"""Dataset sampling: prior over film parameters, forward model, corruption.

Spec §7.1.
Implemented by DTFM-026.

The simulator is the labeller, so training data is unlimited and free and is
generated on the fly rather than stored. There is no dataset on disk to go stale
and no risk of training against one version of the physics while evaluating
against another.

The prior is the substance of this module, not the plumbing. §7.1:

    Sampling design is a decision, not a detail. [...] The prior you sample from
    *is* the model's implicit prior - everything outside it is out-of-
    distribution at test time.

So the choice is recorded in :class:`Prior`, the reasoning in its docstring, and
the support is a stated attribute rather than an implicit consequence of some
sampling code - because §8's calibration work and §10's out-of-distribution
probes both measure against that boundary.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import yaml

from src import dispersion as dp
from src import noise as nz
from src import tmm_torch as pt

__all__ = ["Batch", "Prior", "generate_batch", "load_config", "sample_parameters"]


@dataclass(frozen=True)
class Prior:
    """The distribution training films are drawn from — §7.1's central decision.

    Thickness
    ---------
    ``spacing`` selects uniform or log-uniform sampling, and the two produce
    genuinely different models rather than differently-shaped histograms.

    Over 20-2000 nm, **uniform** places about 95% of samples above 100 nm, where
    fringes are well separated and inversion is easy. **Log-uniform** gives each
    decade equal weight, so a fifth of the samples land below 60 nm.

    The measured argument for log-uniform is that the thin regime is much harder
    than it looks. DTFM-068 found that a film's departure from a bare interface
    falls **quadratically** in thickness on a transparent substrate, not linearly
    — halving an already-thin film quarters the signal. §5.2(c) predicts the fit
    returns a confident meaningless number there, and §8 and §10 exist to
    characterise exactly that. Uniform sampling would leave the network almost no
    examples in the regime the project is about.

    The argument against is equally real: log-uniform spends capacity where the
    answer may be unrecoverable, and the headline RMSE in nanometres looks worse
    because relative error at 30 nm is worth fewer nanometres than at 1500 nm.
    §10 stratifies error by regime anyway, so a single RMSE was never the number
    that mattered.

    Dispersion
    ----------
    Cauchy coefficients ``A`` and ``B``, sampled around a fitted material. Two
    parameters rather than three: §5.2(a) measured the thickness-index
    correlation at 0.95 with two, and every additional coefficient competes for
    the same information in the spectrum (DTFM-025 measured a 556x conditioning
    cost from two extra nuisance parameters).

    Support
    -------
    :attr:`support` states the box explicitly. Everything outside it is
    out-of-distribution by definition, and §10's probes are meaningless without a
    stated boundary.
    """

    thickness_nm: tuple[float, float] = (20.0, 2000.0)
    spacing: str = "log-uniform"
    cauchy_a: tuple[float, float] = (1.40, 1.55)
    cauchy_b: tuple[float, float] = (0.002, 0.012)
    roughness_nm: tuple[float, float] = (0.0, 4.0)
    substrate: str = "Si"

    def __post_init__(self) -> None:
        if self.spacing not in ("uniform", "log-uniform"):
            raise ValueError(f"spacing must be 'uniform' or 'log-uniform', got {self.spacing!r}")
        low, high = self.thickness_nm
        if not 0.0 < low < high:
            raise ValueError(
                f"thickness range must satisfy 0 < low < high, got {self.thickness_nm}"
            )

    @property
    def support(self) -> dict[str, tuple[float, float]]:
        """The box outside which a sample is out-of-distribution, by definition."""
        return {
            "thickness_nm": self.thickness_nm,
            "cauchy_a": self.cauchy_a,
            "cauchy_b": self.cauchy_b,
            "roughness_nm": self.roughness_nm,
        }

    def contains(self, thickness_nm, cauchy_a, cauchy_b) -> np.ndarray:
        """Whether samples lie inside the support. Used by §10's OOD probes."""
        inside = np.ones(np.shape(thickness_nm), dtype=bool)
        for value, (low, high) in (
            (thickness_nm, self.thickness_nm),
            (cauchy_a, self.cauchy_a),
            (cauchy_b, self.cauchy_b),
        ):
            inside &= (np.asarray(value) >= low) & (np.asarray(value) <= high)
        return inside


@dataclass
class Batch:
    """One batch of labelled examples: spectra in, true parameters out."""

    wavelengths_nm: np.ndarray
    spectra: np.ndarray
    thickness_nm: np.ndarray
    cauchy_a: np.ndarray
    cauchy_b: np.ndarray
    roughness_nm: np.ndarray
    clean_spectra: np.ndarray = field(repr=False)

    @property
    def targets(self) -> np.ndarray:
        """``θ`` as an ``(N, 3)`` array — what a network is asked to predict."""
        return np.stack([self.thickness_nm, self.cauchy_a, self.cauchy_b], axis=1)


def sample_parameters(prior: Prior, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw ``count`` films from the prior."""
    low, high = prior.thickness_nm
    if prior.spacing == "log-uniform":
        thickness = np.exp(rng.uniform(np.log(low), np.log(high), count))
    else:
        thickness = rng.uniform(low, high, count)

    return {
        "thickness_nm": thickness,
        "cauchy_a": rng.uniform(*prior.cauchy_a, count),
        "cauchy_b": rng.uniform(*prior.cauchy_b, count),
        "roughness_nm": rng.uniform(*prior.roughness_nm, count),
    }


def generate_batch(
    wavelengths_nm: np.ndarray,
    count: int,
    prior: Prior | None = None,
    corruption: nz.Corruption | None = None,
    rng: np.random.Generator | None = None,
) -> Batch:
    """§7.1's loop: sample θ, run the forward model, corrupt, return the pair.

    The clean spectra are kept alongside the corrupted ones. They are not
    training inputs — a network never sees them — but §7.3's reconstruction loss
    and §10's failure atlas both need to know what the measurement *would* have
    been, and regenerating them later would mean re-running the simulator with
    the same seed and hoping.
    """
    prior = prior or Prior()
    corruption = corruption or nz.Corruption()
    rng = rng or np.random.default_rng()

    parameters = sample_parameters(prior, count, rng)
    substrate_n, substrate_k = dp.load_nk(prior.substrate, wavelengths_nm)
    substrate = torch.tensor(substrate_n + 1j * substrate_k)

    clean = np.empty((count, wavelengths_nm.size))
    observed = np.empty_like(clean)

    for i in range(count):
        film_index = dp.cauchy_n(
            (parameters["cauchy_a"][i], parameters["cauchy_b"][i], 0.0), wavelengths_nm
        )
        per_film = replace(corruption, roughness_nm=float(parameters["roughness_nm"][i]))

        def forward(grid_nm, thickness, _n=film_index, _c=per_film):
            thicknesses, indices = _c.stack_with_defects([float(thickness)], [1.0, _n, substrate])
            return pt.stack_reflectance(
                torch.tensor(np.asarray(grid_nm, dtype=float)), thicknesses, indices, 0.0, "s"
            ).numpy()

        clean[i] = forward(wavelengths_nm, parameters["thickness_nm"][i])
        observed[i] = nz.corrupt(
            wavelengths_nm, forward, float(parameters["thickness_nm"][i]), per_film, rng
        )

    return Batch(
        wavelengths_nm=wavelengths_nm,
        spectra=observed,
        clean_spectra=clean,
        **parameters,
    )


def load_config(path: str | pathlib.Path) -> dict:
    """Read a run configuration. §15: all runs config-driven, nothing implicit."""
    with pathlib.Path(path).open() as handle:
        return yaml.safe_load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate labelled thin-film spectra (§7.1).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    prior = Prior(**config["prior"])
    corruption = nz.Corruption(**config.get("corruption", {}))
    spectrum = config["spectrum"]
    wavelengths = np.linspace(spectrum["low_nm"], spectrum["high_nm"], spectrum["points"])

    count = arguments.count or config["batch"]["count"]
    seed = arguments.seed if arguments.seed is not None else config["batch"]["seed"]
    batch = generate_batch(wavelengths, count, prior, corruption, np.random.default_rng(seed))

    print(f"  prior        {prior.spacing} thickness over {prior.thickness_nm} nm")
    print(f"  support      {prior.support}")
    print(f"  corruption   {', '.join(corruption.active)}")
    print(f"  spectra      {batch.spectra.shape}  seed {seed}")
    print(f"  thickness    {batch.thickness_nm.min():.1f} - {batch.thickness_nm.max():.1f} nm")
    print(f"  reflectance  {batch.spectra.min():.4f} - {batch.spectra.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
