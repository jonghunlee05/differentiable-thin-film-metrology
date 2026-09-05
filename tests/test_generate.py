"""Dataset generation and the sampling prior.

Spec §7.1.
Implemented by DTFM-026.

§7.1 calls the prior "a decision, not a detail", so most of what is asserted here
is about the prior rather than the plumbing: that both spacings are available,
that they differ in the way the decision was made on, that the support is
explicit, and that a run is reproducible from its seed.
"""

import subprocess
import sys

import numpy as np
import pytest

from src import generate as gen
from src import noise as nz

WAVELENGTHS = np.linspace(400.0, 800.0, 60)


def _draw(spacing: str, count: int = 20_000) -> np.ndarray:
    prior = gen.Prior(spacing=spacing)
    return gen.sample_parameters(prior, count, np.random.default_rng(0))["thickness_nm"]


# --- the prior --------------------------------------------------------------


@pytest.mark.parametrize("spacing", ["uniform", "log-uniform"])
def test_samples_stay_inside_the_declared_support(spacing):
    """The support is the definition of out-of-distribution, so it must hold."""
    prior = gen.Prior(spacing=spacing)
    drawn = gen.sample_parameters(prior, 5_000, np.random.default_rng(1))

    assert np.all(prior.contains(drawn["thickness_nm"], drawn["cauchy_a"], drawn["cauchy_b"]))
    for name, (low, high) in prior.support.items():
        assert drawn[name].min() >= low and drawn[name].max() <= high


def test_the_two_spacings_are_genuinely_different_models():
    """§7.1: "Uniform in thickness produces a different model than log-uniform."

    Over 20-2000 nm, uniform places about 95% of samples above 100 nm where
    inversion is easy. Log-uniform gives each decade equal weight and puts a
    third below 100 nm. This is the decision the ticket exists to make, so the
    difference is asserted rather than assumed.
    """
    uniform, log_uniform = _draw("uniform"), _draw("log-uniform")

    assert np.mean(uniform < 100.0) < 0.10
    assert np.mean(log_uniform < 100.0) > 0.25
    assert np.median(log_uniform) < 0.4 * np.median(uniform)


def test_log_uniform_covers_the_regime_the_project_is_about():
    """The measured argument for the default, not a preference.

    Below roughly 50 nm the Cramér-Rao bound on thickness exceeds 0.2% relative
    — the regime §5.2(c) says returns a confident meaningless number, and which
    §8 and §10 exist to characterise. DTFM-068 measured the departure from a bare
    interface as *quadratic* in thickness there, so it is harder than the spec's
    phrasing suggests.

    Uniform sampling puts under 2% of training data there. The network would
    barely see the regime the project is written about.
    """
    hard = 51.0
    assert np.mean(_draw("uniform") < hard) < 0.03
    assert np.mean(_draw("log-uniform") < hard) > 0.15


def test_log_uniform_is_uniform_in_the_logarithm():
    """Each decade gets equal weight — the defining property, not a shape check."""
    drawn = np.log(_draw("log-uniform"))
    edges = np.linspace(drawn.min(), drawn.max(), 6)
    counts = np.histogram(drawn, bins=edges)[0]

    assert counts.max() / counts.min() < 1.15


def test_an_unknown_spacing_is_rejected():
    with pytest.raises(ValueError, match="uniform"):
        gen.Prior(spacing="gaussian")


def test_an_impossible_thickness_range_is_rejected():
    with pytest.raises(ValueError, match="0 < low < high"):
        gen.Prior(thickness_nm=(500.0, 100.0))
    with pytest.raises(ValueError, match="0 < low < high"):
        gen.Prior(thickness_nm=(0.0, 100.0))


def test_the_support_is_stated_rather_than_implied():
    """§10's out-of-distribution probes are meaningless without a boundary."""
    prior = gen.Prior()
    assert set(prior.support) == {"thickness_nm", "cauchy_a", "cauchy_b", "roughness_nm"}

    inside = prior.contains([100.0], [1.46], [0.004])
    outside = prior.contains([100.0], [1.90], [0.004])
    assert bool(inside[0]) and not bool(outside[0])


# --- batches ----------------------------------------------------------------


def test_a_batch_pairs_spectra_with_the_parameters_that_made_them():
    batch = gen.generate_batch(WAVELENGTHS, 8, rng=np.random.default_rng(0))

    assert batch.spectra.shape == (8, WAVELENGTHS.size)
    assert batch.targets.shape == (8, 3)
    assert np.array_equal(batch.targets[:, 0], batch.thickness_nm)
    assert np.all(np.isfinite(batch.spectra))


def test_generation_is_reproducible_from_its_seed():
    """§15: fixed seed, fixed output. Both the sampling and the noise."""
    first = gen.generate_batch(WAVELENGTHS, 6, rng=np.random.default_rng(42))
    second = gen.generate_batch(WAVELENGTHS, 6, rng=np.random.default_rng(42))
    other = gen.generate_batch(WAVELENGTHS, 6, rng=np.random.default_rng(43))

    assert np.array_equal(first.spectra, second.spectra)
    assert np.array_equal(first.thickness_nm, second.thickness_nm)
    assert not np.array_equal(first.spectra, other.spectra)


def test_the_corrupted_spectra_differ_from_the_clean_ones():
    """Otherwise the generator is producing perfect data and proving nothing.

    §4.5's defects are the reason the inverse problem is hard; training on clean
    spectra would give a model that looks excellent and fails on anything real.
    """
    batch = gen.generate_batch(WAVELENGTHS, 8, rng=np.random.default_rng(0))
    difference = np.abs(batch.spectra - batch.clean_spectra)

    assert difference.max() > 1e-4
    assert difference.max() < 0.2, "corruption should not swamp the signal"


def test_the_clean_spectra_are_kept_for_later_tickets():
    """§7.3's reconstruction loss and §10's atlas both need to know what the
    measurement *would* have been. Regenerating them later would mean re-running
    the simulator with the same seed and hoping.
    """
    batch = gen.generate_batch(WAVELENGTHS, 4, rng=np.random.default_rng(0))

    assert batch.clean_spectra.shape == batch.spectra.shape
    assert np.all((batch.clean_spectra >= 0.0) & (batch.clean_spectra <= 1.0))


def test_spectra_stay_physical_across_the_whole_prior():
    """Sampled films span 20-2000 nm and a range of indices and roughness; none
    of them may produce reflectance outside [0, 1] once noise is added.
    """
    batch = gen.generate_batch(WAVELENGTHS, 48, rng=np.random.default_rng(7))

    assert np.all(batch.clean_spectra >= 0.0) and np.all(batch.clean_spectra <= 1.0)
    assert np.all(batch.spectra > -0.05) and np.all(batch.spectra < 1.05)


def test_roughness_is_sampled_per_film_not_fixed():
    """A real wafer's roughness is a property of the film, so it belongs in the
    prior rather than in the instrument configuration.
    """
    batch = gen.generate_batch(WAVELENGTHS, 32, rng=np.random.default_rng(3))

    assert batch.roughness_nm.std() > 0.5
    assert np.all(batch.roughness_nm >= 0.0)


@pytest.mark.parametrize("spacing", ["uniform", "log-uniform"])
def test_both_priors_produce_usable_batches(spacing):
    """Whichever is configured, the generator has to work — the choice is a
    default, not a hard-coding.
    """
    batch = gen.generate_batch(
        WAVELENGTHS, 8, gen.Prior(spacing=spacing), rng=np.random.default_rng(0)
    )

    assert np.all(np.isfinite(batch.spectra))
    assert batch.thickness_nm.min() >= 20.0


# --- configuration and the command line -------------------------------------


def test_the_shipped_config_reproduces_the_documented_default():
    """§15: runs are config-driven. The file is the source of truth for a run,
    so it must agree with what the module documents.
    """
    config = gen.load_config("configs/default.yaml")
    prior = gen.Prior(**config["prior"])

    assert prior.spacing == "log-uniform"
    assert prior.thickness_nm == [20.0, 2000.0]
    assert config["spectrum"]["points"] == 200

    corruption = nz.Corruption(**config["corruption"])
    assert "detector noise" in corruption.active


def test_the_command_line_entry_point_runs():
    """The acceptance criterion, run as a subprocess so it exercises the real
    entry point rather than calling main() directly.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "src.generate", "--config", "configs/default.yaml",
         "--count", "4", "--seed", "1"],
        capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "log-uniform" in completed.stdout
    assert "support" in completed.stdout
    assert completed.stderr == "", f"the run should be warning-free: {completed.stderr}"
