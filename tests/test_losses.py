"""The three loss terms — DTFM-042, spec §7.3.

    L = L_nll + λ_recon · L_recon

§7.3's AC: both terms implemented, and ``L_recon``'s gradients flow through the
transfer matrix. That last clause is the ticket — a reconstruction term that did
not backpropagate would still produce a number, still fall during training, and
still be useless.
"""

import numpy as np
import pytest
import torch

from src import dataset as ds
from src import generate as gen
from src import losses, models

WAVELENGTHS = np.linspace(400.0, 800.0, 200)


@pytest.fixture(scope="module")
def prior():
    return gen.Prior()


@pytest.fixture(scope="module")
def reconstruct(prior):
    return losses.Reconstructor(WAVELENGTHS, prior=prior)


@pytest.fixture(scope="module")
def batch(prior):
    return ds.sample_batch(16, WAVELENGTHS, np.random.default_rng(0), prior=prior)


# --- the acceptance criterion -------------------------------------------------


def test_reconstruction_gradients_flow_through_the_transfer_matrix(batch, reconstruct):
    """§7.3's AC, and the whole point of the term.

    The gradient has to travel from a scalar loss, back through Ψ and Δ, through
    the Airy/transfer-matrix algebra, through the Cauchy dispersion that turns
    two coefficients into an index at every wavelength, and land on the three
    numbers the network predicted. If any link is not differentiable the loss
    still computes and still looks like it is working.

    This is what makes the project scientific ML rather than regression on
    physics-generated data: the simulator is inside the training signal, not
    upstream of it.
    """
    theta = batch.targets.float().clone().requires_grad_(True)

    losses.reconstruction_loss(theta, batch.observed, reconstruct).backward()

    assert theta.grad is not None, "no gradient reached the parameters"
    assert torch.isfinite(theta.grad).all()
    assert (theta.grad.abs() > 0).any(), "a gradient of exactly zero is not a gradient"


def test_the_reconstruction_loss_is_lowest_at_the_true_parameters(batch, reconstruct):
    """The property that makes it a useful training signal rather than a number.

    A film displaced by 5 nm of thickness must reconstruct worse than the true
    film. If it did not, the term would be pulling the network away from the
    answer while appearing to converge.
    """
    truth = batch.targets.float()
    displaced = truth.clone()
    displaced[:, 0] += 5.0

    at_truth = losses.reconstruction_loss(truth, batch.observed, reconstruct)
    at_displaced = losses.reconstruction_loss(displaced, batch.observed, reconstruct)

    assert at_displaced > at_truth * 2, "5 nm of thickness error should be clearly visible"


def test_the_reconstruction_loss_has_a_floor_that_is_not_the_networks_fault(batch, reconstruct):
    """θ̂ has three components; a film has more.

    The network predicts thickness and the two Cauchy coefficients. §4.5 also
    gives every film 0-4 nm of surface roughness, and the observation carries
    instrument noise. Neither is predicted, so ``f(θ̂)`` cannot match an
    observation even when θ̂ is exactly right.

    The floor is reported rather than subtracted. A network told its floor was
    zero would spend capacity chasing a residual made of roughness it was never
    shown, and λ_recon would be tuned against a target that does not exist.
    """
    floor = losses.reconstruction_floor(batch.targets.float(), batch.observed, reconstruct)

    assert floor > 0.0, "with roughness and noise present, a perfect fit is impossible"
    assert torch.isfinite(floor)


# --- the wrap, which would corrupt the loss silently --------------------------


def test_delta_is_compared_on_the_circle_not_the_line():
    """Δ lives on ``(−π, π]``. Crossing the wrap is not a large error.

    A predicted Δ of +179° against an observed −179° is **two degrees** apart.
    Subtracting them gives 358°, and squaring that produces a residual four
    orders of magnitude too large — appearing not at random but on precisely the
    films that sit near the wrap. It would look like the network failing on a
    class of film, and the natural response would be to blame the architecture.
    """
    high = torch.tensor([179.0 * np.pi / 180.0])
    low = torch.tensor([-179.0 * np.pi / 180.0])

    wrapped = losses.wrapped_difference(high, low)
    naive = high - low

    assert abs(float(wrapped)) == pytest.approx(2.0 * np.pi / 180.0, abs=1e-6)
    assert abs(float(naive)) == pytest.approx(358.0 * np.pi / 180.0, abs=1e-6)
    assert float(naive) ** 2 / float(wrapped) ** 2 > 3e4, "the size of the mistake avoided"


def test_the_wrap_is_actually_applied_by_the_reconstruction_loss(reconstruct):
    """The helper being correct is not the same as the loss using it.

    Two observations identical but for Δ shifted by a full turn describe the same
    physical measurement, so the loss must not distinguish them.
    """
    theta = torch.tensor([[400.0, 1.47, 0.006]])
    observed = reconstruct(theta).detach()
    psi, delta = losses.split_psi_delta(observed)
    shifted = torch.cat([psi, delta + 2.0 * np.pi], dim=1)

    assert losses.reconstruction_loss(theta, shifted, reconstruct) == pytest.approx(
        float(losses.reconstruction_loss(theta, observed, reconstruct)), abs=1e-9
    )


def test_a_malformed_observation_is_refused():
    """Ψ and Δ are concatenated, so the width is even. An odd width means the
    caller has passed something that is not an ellipsometry observation, and
    splitting it in half would silently mix the two quantities.
    """
    with pytest.raises(ValueError, match="even width"):
        losses.split_psi_delta(torch.randn(4, 401))


# --- the NLL, and why it trains an uncertainty --------------------------------


def test_the_nll_is_minimised_when_sigma_matches_the_actual_error(prior):
    """The property that makes σ̂ calibrated rather than decorative.

    ``L = ½[e²/σ² + log σ²]``. The first term rewards a large σ̂ — dividing by
    something big makes any error look small. The second punishes it. The minimum
    sits exactly where σ̂² equals the squared error the network really makes, so
    the only way to score well is to state an uncertainty that matches reality.

    This is why MSE could never produce a calibrated σ̂: under MSE the head
    receives no gradient at all.
    """
    scaler = models.ParameterScaler(prior)
    theta = torch.tensor([[400.0, 1.47, 0.006]])
    theta_hat = theta.clone()
    # a known error in cube units, applied to thickness only
    error_cube = 0.02
    theta_hat[0, 0] = theta[0, 0] + error_cube * float(scaler.span[0])

    def score(sigma_cube: float) -> float:
        log_var = torch.full((1, 3), float(np.log(sigma_cube**2)))
        return float(losses.gaussian_nll(theta_hat, log_var, theta, scaler))

    honest = score(error_cube)
    assert honest < score(error_cube * 5), "hedging with a huge sigma must be punished"
    assert honest < score(error_cube / 5), "claiming false confidence must be punished"


def test_the_nll_punishes_confident_wrongness_hardest(prior):
    """Between the two failure modes, overconfidence is the dangerous one.

    A fab acting on a thickness that is wrong *and* marked reliable is worse off
    than one acting on a thickness marked uncertain. The ``e²/σ²`` term grows
    without bound as σ̂ shrinks, while the ``log σ̂²`` penalty for excessive
    hedging grows only logarithmically — the asymmetry is deliberate.
    """
    scaler = models.ParameterScaler(prior)
    theta = torch.tensor([[400.0, 1.47, 0.006]])
    theta_hat = torch.tensor([[500.0, 1.47, 0.006]])  # 100 nm of thickness error

    overconfident = losses.gaussian_nll(theta_hat, torch.full((1, 3), -10.0), theta, scaler)
    hedging = losses.gaussian_nll(theta_hat, torch.full((1, 3), 2.0), theta, scaler)

    assert overconfident > hedging


def test_the_nll_cannot_divide_by_zero(prior):
    """DTFM-041's bounds, exercised through the loss that needed them.

    Nothing in ``e²/σ²`` stops the network driving ``log σ̂²`` towards −∞ on films
    it fits well early in training. Without the clamp this is ``inf`` and the run
    ends rather than converging.
    """
    scaler = models.ParameterScaler(prior)
    theta = torch.tensor([[400.0, 1.47, 0.006]])

    value = losses.gaussian_nll(theta.clone(), torch.full((1, 3), -1e4), theta, scaler)

    assert torch.isfinite(value)


# --- composition --------------------------------------------------------------


def test_the_total_reports_its_parts(batch, prior, reconstruct):
    """With two terms in different units, one number cannot say which one moved.

    DTFM-045 ablates λ_recon by reading exactly these, and a run log that recorded
    only the total would make that ticket impossible after the fact.
    """
    scaler = models.ParameterScaler(prior)
    theta_hat = batch.targets.float().clone()
    log_var = torch.full_like(theta_hat, -4.0)

    total, parts = losses.total_loss(
        theta_hat, batch.targets, batch.observed, scaler,
        log_var=log_var, reconstruct=reconstruct, lambda_recon=0.5,
    )

    assert set(parts) == {"fit", "recon", "total"}
    assert parts["total"] == pytest.approx(parts["fit"] + 0.5 * parts["recon"], rel=1e-5)
    assert torch.isfinite(total)


def test_without_a_head_the_total_falls_back_to_mean_squared_error(batch, prior):
    """The control. Every number in the 141-run history was measured under MSE,
    and dropping it would make everything recorded so far incomparable to
    everything that follows.
    """
    scaler = models.ParameterScaler(prior)
    theta_hat = batch.targets.float().clone()

    total, parts = losses.total_loss(theta_hat, batch.targets, batch.observed, scaler)

    assert "recon" not in parts, "λ_recon defaults to 0, so the term is not computed"
    assert total == pytest.approx(
        float(losses.parameter_loss(theta_hat, batch.targets, scaler)), abs=1e-9
    )


def test_lambda_zero_costs_nothing(batch, prior, reconstruct):
    """λ_recon = 0 must skip the physics entirely, not multiply it by zero.

    The reconstruction term is the expensive part of the step — a full transfer
    matrix over 200 wavelengths per film. Computing it and discarding it would
    make the ablation's control arm as slow as its treatment arm.
    """
    scaler = models.ParameterScaler(prior)
    theta_hat = batch.targets.float().clone().requires_grad_(True)

    _, parts = losses.total_loss(
        theta_hat, batch.targets, batch.observed, scaler,
        reconstruct=reconstruct, lambda_recon=0.0,
    )

    assert "recon" not in parts


def test_the_reconstruction_matches_the_shape_of_a_real_observation(batch, reconstruct):
    """``f(θ̂)`` has to be comparable to what the instrument produced, element for
    element. A mismatch here would broadcast rather than fail.
    """
    predicted = reconstruct(batch.targets.float())

    assert predicted.shape == batch.observed.shape
