"""Learned inversion — the network that replaces the search.

Spec §7.2.
The MLP by DTFM-039; the 1D CNN by DTFM-040; the uncertainty head by DTFM-041.

§7.2 is explicit about scope:

    **Do not reach for a transformer.** Architecture novelty is not what is being
    assessed here, and over-reaching signals inexperience rather than ambition.

So this is a plain regression: 400 numbers in, 3 out. What is interesting about
this project is not the architecture — it is whether the resulting estimator can
say when it is wrong, and that lives in DTFM-041's second head and in E6's
calibration work.

The two scalings, which are not cosmetic
----------------------------------------
Both are here because DTFM-031 measured the cost of omitting them.

**Inputs are standardised.** Ψ spans roughly 0-90° and Δ spans ±180°, so the raw
vector hands the network one signal an order of magnitude larger than the other.
The scaling is fixed at construction from the observable's known range rather
than estimated from a batch, because a running estimate would make the network's
behaviour depend on which films it happened to see first.

**Outputs live on the prior's unit cube.** Thickness is ~500 and Cauchy B is
~0.004 — five orders of magnitude apart. DTFM-031 found that a single optimiser
step size is either negligible for one or divergent for the other, and the same
argument applies to a loss summed over parameters: without scaling, thickness
error dominates and the dispersion coefficients are effectively unlearned.

Predicting on the cube and mapping back also makes every prediction physical by
construction — a sigmoid cannot leave the prior's support, so the network cannot
return a negative thickness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src import generate as gen

__all__ = ["InverseMLP", "ParameterScaler", "SpectrumScaler", "build_model"]


@dataclass
class SpectrumScaler:
    """Fixed standardisation of the observation vector.

    Constants rather than batch statistics. §15 wants a run reproducible from its
    config, and a scaler fitted to whatever films arrived first is a hidden
    dependency on the data order — two runs from the same config would differ.

    Ψ lies in ``[0, π/2]`` and Δ in ``(−π, π]`` by construction, so the ranges are
    known exactly and need not be estimated at all.
    """

    observable: str = "ellipsometry"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.observable != "ellipsometry":
            return (x - 0.5) * 4.0  # reflectance is already in [0, 1]
        half = x.shape[-1] // 2
        psi = (x[..., :half] - np.pi / 4) / (np.pi / 4)
        delta = x[..., half:] / np.pi
        return torch.cat([psi, delta], dim=-1)


@dataclass
class ParameterScaler:
    """Maps ``θ`` onto (a margin around) the prior's unit cube, and back.

    The network predicts in cube coordinates; :meth:`decode` returns nanometres
    and dispersion coefficients. Because ``decode`` is applied after a sigmoid,
    **no prediction can fall outside the prior** — the network cannot return a
    negative thickness, and §10's out-of-distribution work therefore tests the
    network's *estimates* rather than its arithmetic.
    """

    prior: gen.Prior
    margin: float = 0.0

    def __post_init__(self) -> None:
        low = np.array([self.prior.thickness_nm[0], self.prior.cauchy_a[0], self.prior.cauchy_b[0]])
        span = np.array(
            [
                self.prior.thickness_nm[1] - self.prior.thickness_nm[0],
                self.prior.cauchy_a[1] - self.prior.cauchy_a[0],
                self.prior.cauchy_b[1] - self.prior.cauchy_b[0],
            ]
        )
        # `margin` widens the sigmoid's range beyond the prior. A sigmoid is flat
        # at its ends, so a film at the very edge of the prior needs a large
        # pre-activation to reach it and learns slowly there — and §7.1's
        # log-uniform prior puts a great many films near those edges. Widening
        # moves the flat region outside the range the data actually occupies.
        #
        # It trades a guarantee for a gradient. At margin=0 no prediction can
        # leave the prior at all; at margin=0.2 a prediction can fall up to 20%
        # outside it, which is physically impossible for a thickness and must be
        # reported rather than silently clipped — see `outside_prior`.
        self.low = torch.tensor(low - self.margin * span)
        self.span = torch.tensor(span * (1.0 + 2.0 * self.margin))
        self._prior_low = torch.tensor(low)
        self._prior_high = torch.tensor(low + span)

    def outside_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Which predictions fell outside the prior — impossible, if it happens.

        With ``margin > 0`` this can be non-empty. It is measured and reported
        rather than clipped away, because a network predicting a negative
        thickness is telling you something about itself.
        """
        return ((theta < self._prior_low) | (theta > self._prior_high)).any(dim=-1)

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.low) / self.span

    def decode(self, unit: torch.Tensor) -> torch.Tensor:
        return self.low + unit * self.span


class InverseMLP(nn.Module):
    """§7.2's first architecture: spectrum in, parameters out.

    Deliberately the simplest thing that could work. It treats the 400 inputs as
    an unordered vector and has to *learn* that neighbouring wavelengths are
    related — which is exactly the weakness DTFM-040's convolutional model is
    meant to fix, and the reason both are built rather than only the better one.

    ``uncertainty=True`` adds DTFM-041's second head. It is wired here rather
    than bolted on later so that the two architectures cannot diverge in how they
    express it, but the head is unused until DTFM-042 supplies a loss that trains
    it — a ``log σ̂²`` output with no NLL term attached is an untrained number, and
    reporting it would be worse than not having it.
    """

    def __init__(
        self,
        input_size: int = 400,
        width: int = 256,
        depth: int = 3,
        *,
        uncertainty: bool = False,
        prior: gen.Prior | None = None,
        observable: str = "ellipsometry",
        output_margin: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")

        self.scale_x = SpectrumScaler(observable)
        self.scale_theta = ParameterScaler(prior or gen.Prior(), margin=output_margin)
        self.uncertainty = uncertainty

        layers: list[nn.Module] = []
        size = input_size
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.trunk = nn.Sequential(*layers)
        self.head_theta = nn.Linear(size, 3)
        self.head_log_var = nn.Linear(size, 3) if uncertainty else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Returns ``θ̂`` in physical units, or ``(θ̂, log σ̂²)`` with the head on.

        ``log σ̂²`` is returned in **cube** units, matching the space the network
        predicts in. Converting it to nanometres is DTFM-041's problem, and doing
        it here would bake in a choice before the loss that uses it exists.
        """
        features = self.trunk(self.scale_x(x))
        theta = self.scale_theta.decode(torch.sigmoid(self.head_theta(features)))
        if self.head_log_var is None:
            return theta
        return theta, self.head_log_var(features)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(config: dict | None = None, **overrides):
    """Build a model from a config dictionary — §7.2's AC.

    "architecture and hyperparameters in config" is a reproducibility requirement,
    not tidiness: §15 asks that the run behind a reported number be reproducible
    from its config, which is impossible if the architecture lives in a script.
    """
    settings = {
        "architecture": "mlp",
        "input_size": 400,
        "width": 256,
        "depth": 3,
        "uncertainty": False,
        "observable": "ellipsometry",
        "output_margin": 0.0,
    }
    settings.update(config or {})
    settings.update(overrides)

    architecture = settings.pop("architecture")
    prior = settings.pop("prior", None)
    if architecture != "mlp":
        raise ValueError(f"unknown architecture {architecture!r}; only 'mlp' exists until DTFM-040")
    return InverseMLP(prior=prior, **settings)
