"""Parameter loss, physics reconstruction loss, Gaussian NLL — DTFM-042, §7.3.

    L = L_nll + λ_recon · L_recon

The second term is what separates this project from generic regression. Without
it a network learns the map from spectra to parameters the way it would learn any
table of numbers: the physics only ever generated the data. With it, the
network's own guess is pushed **back** through the differentiable transfer-matrix
model and compared against the spectrum it started from, so the simulator becomes
part of the training signal rather than a fixture that ran beforehand.

Everything here computes in **cube units** for the parameter terms (§7.1's unit
box, where thickness, ``A`` and ``B`` are comparable) and in **radians** for the
reconstruction term (where the instrument's noise is defined). Mixing those was
the failure recorded in Implementation-Notes §28, and the note applies here with
more force: a loss that silently changes units between experiments makes every
comparison between them meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from . import dispersion as dp
from . import generate as gen
from . import models
from . import tmm_torch as pt


def split_psi_delta(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flattened ``(batch, 2W)`` ellipsometry observation into ``(Ψ, Δ)``.

    The layout is ``cat([psi, delta])`` over the wavelength grid, so an
    observation of width ``2W`` splits at ``W``. Written once because slicing it
    by hand at each call site is how Ψ and Δ eventually get swapped — and a
    swapped Ψ/Δ would not crash, it would train.
    """
    if observed.shape[-1] % 2:
        raise ValueError(
            f"an ellipsometry observation has an even width (Ψ then Δ), got {observed.shape[-1]}"
        )
    half = observed.shape[-1] // 2
    return observed[..., :half], observed[..., half:]


def wrapped_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a − b`` for angles that live on a circle, mapped back into ``(−π, π]``.

    **This is the trap in the reconstruction loss.** Δ wraps: a predicted +179°
    and an observed −179° are two degrees apart, and a plain difference calls it
    358°. Squared, that is a residual four orders of magnitude too large, and it
    appears precisely on the films sitting near the wrap rather than at random —
    so it would not look like noise, it would look like the network failing on a
    particular class of film, and the natural response would be to blame the
    architecture.

    ``dataset.py`` already refuses to smooth Δ directly for the same reason.
    """
    difference = a - b
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def parameter_loss(
    theta_hat: torch.Tensor, theta: torch.Tensor, scaler: models.ParameterScaler
) -> torch.Tensor:
    """Mean squared error in cube units — the loss DTFM-039 through DTFM-070 used.

    Kept because it is the control. Every number in the run history was measured
    under it, and replacing it without keeping it available would make the 141
    runs already recorded incomparable to anything that follows.
    """
    return torch.nn.functional.mse_loss(scaler.encode(theta_hat), scaler.encode(theta).float())


def gaussian_nll(
    theta_hat: torch.Tensor,
    log_var: torch.Tensor,
    theta: torch.Tensor,
    scaler: models.ParameterScaler,
) -> torch.Tensor:
    """§7.3's ``L_nll = ½ Σ_k [ (θ̂_k − θ_k)² / σ̂_k² + log σ̂_k² ]``.

    **Why this trains an uncertainty at all**, which is not obvious from the
    formula. The first term rewards a large σ̂: dividing by something big makes
    any error look small. The second punishes it. The minimum of the two sits
    where σ̂² equals the squared error the network actually makes — so the only
    way to score well is to state an uncertainty that matches reality. Claiming
    confidence you do not have is punished by the first term; hedging everything
    is punished by the second.

    That is the whole reason DTFM-041's head is worth having, and the reason MSE
    could never produce a calibrated σ̂: under MSE the head receives no gradient
    at all.

    Computed in cube units, matching the space the head emits ``log σ̂²`` in.
    Converting to nanometres of thickness here would make the three parameters
    incommensurable again — the thickness term would outweigh ``B`` by ten orders
    of magnitude, which is DTFM-031's finding applied to a loss.
    """
    bounded = models.clamp_log_var(log_var)
    error = scaler.encode(theta_hat) - scaler.encode(theta).float()
    return 0.5 * (error**2 / torch.exp(bounded) + bounded).mean()


@dataclass
class Reconstructor:
    """Runs the forward model on a *predicted* θ̂, differentiably.

    Separate from :func:`dataset.sample_batch` for one reason that matters:
    ``sample_batch`` wraps its physics in ``torch.no_grad()`` because generating
    training data needs no gradient. Here the gradient **is** the point — §7.3's
    AC is that ``L_recon`` flows back through the transfer matrix — so the same
    physics has to be re-entered with autograd live.

    **What this cannot reproduce, by construction.** The network predicts three
    parameters: thickness, and the Cauchy coefficients ``A`` and ``B``. A real
    film in this project also has a *roughness* (§4.5 draws 0–4 nm of it) and the
    observation carries instrument noise. Neither is predicted, so ``f(θ̂)``
    cannot match an observation even when θ̂ is exactly right.

    ``L_recon`` therefore has a floor that is not the network's fault, made of
    unmodelled roughness plus noise. :func:`reconstruction_floor` measures it, and
    it is the number λ_recon has to be read against — a reconstruction loss
    sitting at the floor means the term has done everything it can, not that
    training failed.
    """

    wavelengths_nm: NDArray
    measurement: gen.Measurement = field(default_factory=gen.Measurement)
    prior: gen.Prior = field(default_factory=gen.Prior)

    def __post_init__(self) -> None:
        self.wavelengths_nm = np.asarray(self.wavelengths_nm, dtype=float)
        n, k = dp.load_nk(self.prior.substrate, self.wavelengths_nm)
        # Substrate and grid are fixed by the instrument, not by θ̂, so they are
        # built once. Reloading the nk table every training step would dominate
        # the cost of the term it serves.
        self._substrate = torch.tensor(n + 1j * k)[None, :]
        self._grid = torch.as_tensor(self.wavelengths_nm)[None, :]

    def __call__(self, theta_hat: torch.Tensor) -> torch.Tensor:
        """``θ̂ → (Ψ, Δ)`` flattened to ``(batch, 2W)``, with gradients attached."""
        thickness = theta_hat[:, 0:1]
        cauchy_a = theta_hat[:, 1:2]
        cauchy_b = theta_hat[:, 2:3]

        film_index = dp.cauchy_n((cauchy_a, cauchy_b, torch.zeros(())), self._grid)
        layers = [thickness]
        media = [torch.ones(()), film_index, self._substrate]

        psi, delta = pt.stack_psi_delta(
            self._grid, layers, media, self.measurement.angle_rad
        )
        return torch.cat([psi, delta], dim=1)


def reconstruction_loss(
    theta_hat: torch.Tensor, observed: torch.Tensor, reconstruct: Reconstructor
) -> torch.Tensor:
    """§7.3's ``L_recon = ‖f(θ̂) − R_obs‖²``, with Δ compared on the circle.

    Ψ and Δ are both in radians and both carry the same instrument noise, so they
    are weighted equally and no scaling constant is introduced — a weight here
    would be a second λ nobody had agreed to.
    """
    predicted_psi, predicted_delta = split_psi_delta(reconstruct(theta_hat))
    observed_psi, observed_delta = split_psi_delta(observed)

    psi_residual = predicted_psi - observed_psi.to(predicted_psi.dtype)
    delta_residual = wrapped_difference(predicted_delta, observed_delta.to(predicted_delta.dtype))
    return (psi_residual**2).mean() + (delta_residual**2).mean()


def reconstruction_floor(
    theta: torch.Tensor, observed: torch.Tensor, reconstruct: Reconstructor
) -> torch.Tensor:
    """The reconstruction loss at the *true* parameters — the best score possible.

    Everything left is unmodelled roughness and instrument noise. Reported rather
    than subtracted: a network whose ``L_recon`` sits here has extracted all the
    physics available to it, and one told that its floor is zero would keep
    chasing a residual it cannot remove.
    """
    with torch.no_grad():
        return reconstruction_loss(theta, observed, reconstruct)


def total_loss(
    theta_hat: torch.Tensor,
    theta: torch.Tensor,
    observed: torch.Tensor,
    scaler: models.ParameterScaler,
    *,
    log_var: torch.Tensor | None = None,
    reconstruct: Reconstructor | None = None,
    lambda_recon: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``L = L_nll + λ_recon · L_recon``, or MSE when no head is supplied.

    Returns the loss and its parts. The parts are not decoration: with two terms
    in different units a single total cannot say which one moved, and DTFM-045
    ablates λ_recon by reading exactly these.
    """
    fit = (
        parameter_loss(theta_hat, theta, scaler)
        if log_var is None
        else gaussian_nll(theta_hat, log_var, theta, scaler)
    )
    parts = {"fit": float(fit.detach())}

    total = fit
    if lambda_recon and reconstruct is not None:
        recon = reconstruction_loss(theta_hat, observed, reconstruct)
        total = total + lambda_recon * recon
        parts["recon"] = float(recon.detach())
    parts["total"] = float(total.detach())
    return total, parts
