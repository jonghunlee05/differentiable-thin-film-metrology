"""The learned inverse model — §7.2.

Implemented by DTFM-039.

These pin the model's *contract*, not its accuracy. Accuracy is a training
outcome that changes with every hyperparameter; what must not change is that the
model consumes what the pipeline produces, emits something physical, and can be
rebuilt from a config.
"""

import numpy as np
import pytest
import torch

from src import dataset as ds
from src import generate as gen
from src import models

WAVELENGTHS = np.linspace(400.0, 800.0, 200)


def test_it_consumes_what_the_pipeline_produces():
    """The join between DTFM-038 and DTFM-039, asserted rather than assumed.

    A shape mismatch here would only surface as a crash mid-training, and a
    silent broadcast would be worse — it would train on nonsense and report a
    number.
    """
    batch = ds.sample_batch(8, WAVELENGTHS, np.random.default_rng(0))
    model = models.build_model()

    predicted = model(batch.observed.float())
    assert predicted.shape == batch.targets.shape == (8, 3)
    assert torch.isfinite(predicted).all()


def test_no_prediction_can_leave_the_prior():
    """The sigmoid's whole purpose, at the default ``output_margin = 0``.

    A negative thickness is not a large error, it is an impossible one. Ruling it
    out by construction means §10's out-of-distribution work tests the model's
    *estimates* rather than its arithmetic.
    """
    prior = gen.Prior()
    model = models.build_model(prior=prior)

    absurd = model(torch.randn(256, 400) * 100)
    assert (absurd[:, 0] >= prior.thickness_nm[0]).all()
    assert (absurd[:, 0] <= prior.thickness_nm[1]).all()
    assert not model.scale_theta.outside_prior(absurd).any()


def test_a_margin_trades_that_guarantee_for_a_gradient():
    """``output_margin`` widens the sigmoid past the prior, and the cost is real.

    Measured across runs 1 and 2 at margin 0.2: the training loss improved 2.6x,
    the median thickness error got **7% worse**, and 4.3% of predictions landed
    outside the prior — physically impossible thicknesses.

    The option stays because that comparison is worth being able to re-run. The
    default stays at 0 because the guarantee is worth more than a loss number,
    and because the loss is only about half thickness: it improved on the other
    half.
    """
    prior = gen.Prior()
    widened = models.build_model({"output_margin": 0.2}, prior=prior)

    absurd = widened(torch.randn(256, 400) * 100)
    assert widened.scale_theta.outside_prior(absurd).any(), (
        "a margin must actually allow predictions outside the prior, or it does nothing"
    )
    assert absurd[:, 0].min() < prior.thickness_nm[0]


def test_the_input_scaling_uses_known_ranges_not_batch_statistics():
    """§15 wants a run reproducible from its config.

    A scaler fitted to whatever films arrived first is a hidden dependency on data
    order — two runs from one config would differ. Ψ lies in [0, π/2] and Δ in
    (−π, π] by construction, so the ranges are known exactly and need not be
    estimated at all.
    """
    model = models.build_model()
    first = ds.sample_batch(64, WAVELENGTHS, np.random.default_rng(0))
    second = ds.sample_batch(64, WAVELENGTHS, np.random.default_rng(1))

    for batch in (first, second):
        scaled = model.scale_x(batch.observed.float())
        assert scaled.abs().max() < 3.0, "roughly unit scale for any batch"

    # the scaler holds no state, so it cannot drift with the data it has seen
    assert not list(models.SpectrumScaler().__dict__.get("_fitted", []) or [])


def test_the_parameter_scaling_round_trips():
    prior = gen.Prior()
    scaler = models.ParameterScaler(prior)
    theta = torch.tensor([[420.0, 1.46, 0.004], [20.0, 1.40, 0.002]], dtype=torch.float64)

    assert torch.allclose(scaler.decode(scaler.encode(theta)), theta, atol=1e-9)
    assert scaler.encode(theta).min() >= 0.0
    assert scaler.encode(theta).max() <= 1.0


def test_the_model_is_reproducible_from_its_config():
    """§7.2's AC — "architecture and hyperparameters in config"."""
    config = {"width": 128, "depth": 2, "output_margin": 0.1}
    torch.manual_seed(0)
    first = models.build_model(config)
    torch.manual_seed(0)
    second = models.build_model(config)

    assert first.parameter_count == second.parameter_count
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_uncertainty_head_is_wired_but_off_by_default():
    """DTFM-041's second head exists so the architectures cannot diverge in how
    they express it — but it is unused until DTFM-042 supplies an NLL loss.

    A ``log σ̂²`` output with nothing training it is an untrained number, and
    reporting one would be worse than not having it.
    """
    plain = models.build_model()
    assert isinstance(plain(torch.randn(4, 400)), torch.Tensor)

    with_head = models.build_model({"uncertainty": True})
    theta, log_var = with_head(torch.randn(4, 400))
    assert theta.shape == log_var.shape == (4, 3)
    assert with_head.parameter_count > plain.parameter_count


@pytest.mark.parametrize(("width", "depth"), [(0, 3), (256, 0), (-1, 2)])
def test_a_degenerate_architecture_is_refused(width, depth):
    with pytest.raises(ValueError):
        models.build_model({"width": width, "depth": depth})


def test_an_unknown_architecture_is_refused():
    with pytest.raises(ValueError, match="architecture"):
        models.build_model({"architecture": "transformer"})


def test_it_learns_something_in_a_few_hundred_steps():
    """§7.2's AC — "trains and beats a trivial baseline".

    Deliberately a weak, fast check: 400 steps is nowhere near convergence. The
    real numbers live in ``runs/history.jsonl`` and are reported per run. What
    this asserts is that the loop *learns at all* — a model wired up wrongly
    (frozen weights, detached graph, shuffled targets) would sit flat here.

    The trivial baseline is "always guess the geometric mean of the prior", which
    scores about 164 nm. Linear regression on the same data scores 154 nm, so
    even a straight-line model barely beats guessing — the relationship is
    strongly nonlinear, and that is what makes the network worth having.
    """
    torch.manual_seed(0)
    prior = gen.Prior()
    model = models.build_model(prior=prior)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)

    first = last = None
    for step in range(400):
        batch = ds.sample_batch(64, WAVELENGTHS, rng, prior=prior)
        optimiser.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model.scale_theta.encode(model(batch.observed.float())),
            model.scale_theta.encode(batch.targets).float(),
        )
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < 0.5 * first, f"loss barely moved: {first:.4f} -> {last:.4f}"


# --- DTFM-040: the convolutional model ----------------------------------------


def test_the_cnn_reshapes_the_spectrum_into_wavelength_and_channel():
    """The load-bearing decision in :class:`models.InverseCNN`.

    The 400 inputs are **200 wavelengths × 2 channels**, not a flat run of 400.
    ``Ψ[i]`` and ``Δ[i]`` are the same wavelength — they belong at the same
    position in different channels, exactly as red and green are one pixel.

    A ``1 × 400`` layout would let the kernel straddle index 200, mixing the phase
    near 800 nm with the amplitude near 400 nm. That is physically meaningless and
    it would train perfectly happily while doing it — the model would simply be
    worse, with nothing to say why. Asserted because nothing else would catch it.
    """
    model = models.build_model({"architecture": "cnn"})

    assert model.channels_in == 2, "Ψ and Δ are channels, not a concatenated sequence"
    batch = ds.sample_batch(4, WAVELENGTHS, np.random.default_rng(0))
    stacked = model.scale_x(batch.observed.float()).reshape(4, model.channels_in, -1)
    assert stacked.shape == (4, 2, 200)

    # channel 0 must be Ψ and channel 1 Δ, in the pipeline's own order
    half = batch.observed.shape[1] // 2
    assert torch.allclose(stacked[:, 0, :], model.scale_x(batch.observed.float())[:, :half])


def test_the_cnn_wins_on_structure_not_on_size():
    """§7.2 predicts the CNN is "better suited — fringes are local repeating
    structure". DTFM-040 measured it, and the parameter count is what makes the
    result attributable:

    | | MLP | CNN |
    |---|---|---|
    | parameters | 235,011 | **117,795** |
    | median error | 5.977 nm | **4.758 nm** |
    | thin films | 4.453 nm | **2.585 nm** |

    20% better overall with **half** the parameters, and 42% better on thin films
    — where fringes are few and widely spaced, which is exactly the local pattern
    a convolution is built to detect. On thick films, where fringes crowd the band
    and the signal is closer to a global property, the advantage vanishes.

    This test pins the parameter relationship only. Accuracy is a training outcome
    and lives in ``runs/history.jsonl``; what must stay true is that the CNN is not
    quietly winning by being bigger.
    """
    mlp = models.build_model({"architecture": "mlp"})
    cnn = models.build_model({"architecture": "cnn"})

    assert cnn.parameter_count < mlp.parameter_count, (
        "the comparison is only meaningful while the CNN is the smaller model"
    )


def test_both_architectures_satisfy_the_same_contract():
    """DTFM-041's head and DTFM-044's benchmark treat these interchangeably, so
    they must agree on everything but their internals.
    """
    batch = ds.sample_batch(6, WAVELENGTHS, np.random.default_rng(0))
    prior = gen.Prior()

    for architecture in ("mlp", "cnn"):
        model = models.build_model({"architecture": architecture}, prior=prior)
        predicted = model(batch.observed.float())
        assert predicted.shape == (6, 3), architecture
        assert not model.scale_theta.outside_prior(predicted).any(), architecture

        with_head = models.build_model(
            {"architecture": architecture, "uncertainty": True}, prior=prior
        )
        theta, log_var = with_head(batch.observed.float())
        assert theta.shape == log_var.shape == (6, 3), architecture


@pytest.mark.parametrize("kernel", [2, 4, 8])
def test_an_even_kernel_is_refused(kernel):
    """Padding is ``kernel // 2`` on both sides, which only centres the kernel when
    it is odd. An even kernel would shift the output by half a sample per layer —
    a systematic wavelength offset that would look like a calibration error.
    """
    with pytest.raises(ValueError, match="odd"):
        models.build_model({"architecture": "cnn", "kernel": kernel})


def test_the_cnn_is_reproducible_from_its_config():
    config = {"architecture": "cnn", "channels": 16, "depth": 2, "kernel": 5}
    torch.manual_seed(0)
    first = models.build_model(config)
    torch.manual_seed(0)
    second = models.build_model(config)

    assert first.parameter_count == second.parameter_count
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


# --- DTFM-041, the heteroscedastic head ---------------------------------------
#
# §7.2, §8.1. The head itself was wired during DTFM-039/040 so the two
# architectures could not diverge in how they express it. What follows is the
# rest: getting σ̂ out in units anyone can act on, and keeping the arithmetic
# finite before DTFM-042's NLL starts dividing by it.


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_sigma_comes_back_in_nanometres_not_cube_units(architecture):
    """The conversion this ticket exists for, checked against the arithmetic.

    The head emits ``log σ̂²`` in cube coordinates. A σ̂ of 0.001 there is 1.98 nm,
    not 0.001 nm — a factor of 1980. An error of that size would not look like an
    error: a picometre uncertainty is precisely what a hopeful reader expects to
    see on a metrology result, and it would be reported rather than questioned.
    """
    prior = gen.Prior()
    model = models.build_model({"architecture": architecture, "uncertainty": True}, prior=prior)

    _, log_var = model(torch.randn(8, 400))
    _, sigma = model.predict(torch.randn(8, 400))

    span = model.scale_theta.span
    assert torch.allclose(
        model.scale_theta.sigma_from_log_var(log_var),
        torch.exp(0.5 * models.clamp_log_var(log_var)) * span,
    )
    assert span[0].item() == pytest.approx(1980.0), "prior thickness span, 20-2000 nm"
    assert (sigma > 0).all(), "a standard deviation is positive by construction"
    assert torch.isfinite(sigma).all()


def test_converting_sigma_uses_the_scale_and_not_the_offset():
    """``decode`` is ``low + u·span``; a spread is a width, so only ``span`` acts.

    Applying the full affine map to a standard deviation would add the prior's
    lower bound — 20 nm for thickness — to every uncertainty, giving a model that
    can never claim to know anything better than 20 nm however well it fits.
    """
    scaler = models.ParameterScaler(gen.Prior())
    sigma_cube = torch.tensor([[0.001, 0.001, 0.001]])

    physical = scaler.decode_sigma(sigma_cube)

    assert physical[0, 0].item() == pytest.approx(1.98), "0.001 x 1980 nm"
    assert not torch.allclose(physical, scaler.decode(sigma_cube)), "decode would add `low`"


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_a_model_without_the_head_reports_no_uncertainty_rather_than_zero(architecture):
    """``None``, not zeros. A σ̂ of zero is a claim of perfect confidence, and a
    model that was never given the head has not made that claim. Returning zeros
    would let it flow silently into a calibration plot as infinite precision.
    """
    model = models.build_model({"architecture": architecture}, prior=gen.Prior())

    theta, sigma = model.predict(torch.randn(4, 400))

    assert theta.shape == (4, 3)
    assert sigma is None


def test_log_variance_is_bounded_so_the_nll_cannot_divide_by_nothing():
    """DTFM-042's NLL contains ``(θ̂ − θ)² / σ̂²``.

    Nothing in that expression stops the network from driving ``log σ̂²`` towards
    −∞ on films it fits well early in training. The local gradient points that
    way, σ̂² underflows to zero, and the run ends in a NaN instead of converging.
    The bound is what makes the loss safe to write in the first place.
    """
    extreme = torch.tensor([[-500.0, 0.0, 500.0]])

    clamped = models.clamp_log_var(extreme)
    sigma = models.ParameterScaler(gen.Prior()).sigma_from_log_var(extreme)

    low, high = models.LOG_VAR_BOUNDS
    assert clamped[0, 0].item() == low
    assert clamped[0, 2].item() == high
    assert (sigma > 0).all() and torch.isfinite(sigma).all()
    assert torch.isfinite(1.0 / sigma**2).all(), "the term the NLL actually computes"


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_sigma_varies_with_the_input(architecture):
    """§8.1's requirement: the uncertainty is *heteroscedastic* — it depends on the
    film being measured, not one number shared by every prediction.

    That is the whole point. Some films are genuinely ambiguous — a thicker film
    with a lower index produces nearly the spectrum of a thinner one with a
    higher index — and a single global error bar could not say so.

    NOTE what this does and does not show. It shows the head is *wired* to the
    input rather than to a bias term: different spectra, different σ̂. It does not
    show that σ̂ is **correct**, because nothing has trained it yet. That
    demonstration needs DTFM-042's NLL and belongs there; an untrained head that
    varies with input is varying the way any random linear layer varies.
    """
    prior = gen.Prior()
    torch.manual_seed(0)
    model = models.build_model({"architecture": architecture, "uncertainty": True}, prior=prior)
    batch = ds.sample_batch(32, WAVELENGTHS, np.random.default_rng(0), prior=prior)

    _, sigma = model.predict(batch.observed.float())

    thickness_sigma = sigma[:, 0]
    assert thickness_sigma.std() > 0, "a constant σ̂ would not be heteroscedastic"
    assert thickness_sigma.min() != thickness_sigma.max()


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_an_untrained_head_admits_it_knows_nothing(architecture):
    """A sanity check on the units, expressed as a physical statement.

    At initialisation the head's output sits near zero, so σ̂ in cube units is
    near 1 — and 1 in cube units is the entire prior. An untrained model should
    therefore report an uncertainty of roughly the full 20-2000 nm range, which
    is the correct thing for it to say. If the conversion were wrong by the 1980x
    factor this test would instead see about a nanometre, which would look like a
    superbly calibrated network that had never been trained.
    """
    torch.manual_seed(0)
    model = models.build_model(
        {"architecture": architecture, "uncertainty": True}, prior=gen.Prior()
    )

    _, sigma = model.predict(torch.randn(16, 400))

    assert sigma[:, 0].median() > 500.0, "an untrained head should claim near-total ignorance"


# --- DTFM-047: the deep ensemble ----------------------------------------------


def _members(count, prior, uncertainty=True):
    made = []
    for seed in range(count):
        torch.manual_seed(seed)
        made.append(
            models.build_model(
                {"architecture": "mlp", "width": 32, "depth": 1, "uncertainty": uncertainty},
                prior=prior,
            )
        )
    return made


def test_the_two_uncertainties_add_in_quadrature():
    """§8.1's formula, checked as an identity rather than trusted.

    ``var = (1/M) Σ (σ̂² + θ̂²) − mean²`` splits exactly into the mean of the
    members' variances plus the variance of their means. If the split did not
    close, the two columns would not be two parts of one number and reporting
    them separately would be meaningless.
    """
    prior = gen.Prior()
    ensemble = models.Ensemble(_members(4, prior))

    parts = ensemble.decompose(torch.randn(6, 400))

    assert torch.allclose(
        parts["total"] ** 2, parts["aleatoric"] ** 2 + parts["epistemic"] ** 2, atol=1e-6
    )


def test_identical_members_disagree_about_nothing():
    """Epistemic uncertainty is disagreement. Five copies of one network have
    none, and must report exactly zero rather than a small positive number —
    a floating-point residue here would read as real model uncertainty.
    """
    prior = gen.Prior()
    torch.manual_seed(0)
    one = models.build_model(
        {"architecture": "mlp", "width": 32, "depth": 1, "uncertainty": True}, prior=prior
    )

    parts = models.Ensemble([one, one, one]).decompose(torch.randn(4, 400))

    assert torch.allclose(parts["epistemic"], torch.zeros_like(parts["epistemic"]), atol=1e-6)
    assert torch.allclose(parts["total"], parts["aleatoric"], atol=1e-6)


def test_different_seeds_do_disagree():
    """The converse. Members trained from different seeds land in different
    places, and that spread is the signal a single network cannot produce about
    itself — a confidently wrong model has siblings that say otherwise.
    """
    prior = gen.Prior()
    ensemble = models.Ensemble(_members(5, prior))

    parts = ensemble.decompose(torch.randn(8, 400))

    assert (parts["epistemic"] > 0).any()


def test_an_ensemble_is_a_drop_in_for_a_single_model():
    """``predict`` has the same signature, so DTFM-044's benchmark and DTFM-048's
    calibration work talk to five networks without knowing it. A separate API
    would mean a separate code path, and a separate code path is where the two
    quietly stop being compared on the same terms.
    """
    prior = gen.Prior()
    ensemble = models.Ensemble(_members(3, prior))
    x = torch.randn(5, 400)

    theta, sigma = ensemble.predict(x)

    assert theta.shape == sigma.shape == (5, 3)
    assert (sigma > 0).all()
    assert torch.allclose(theta, ensemble.decompose(x)["mean"])


def test_members_without_the_head_are_refused():
    """An ensemble of heads-off models can report disagreement but not ambiguity.
    §8.1 asks for both parts, and returning epistemic alone under the name
    "uncertainty" would understate it on exactly the films that are genuinely
    hard to measure.
    """
    prior = gen.Prior()
    ensemble = models.Ensemble(_members(2, prior, uncertainty=False))

    with pytest.raises(ValueError, match="uncertainty head"):
        ensemble.decompose(torch.randn(3, 400))


def test_an_empty_ensemble_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        models.Ensemble([])


def test_the_ensemble_estimate_is_the_mean_of_its_members():
    prior = gen.Prior()
    members = _members(4, prior)
    x = torch.randn(6, 400)

    mean = models.Ensemble(members).decompose(x)["mean"]

    expected = torch.stack([m.predict(x)[0] for m in members]).mean(dim=0)
    assert torch.allclose(mean, expected, atol=1e-6)
