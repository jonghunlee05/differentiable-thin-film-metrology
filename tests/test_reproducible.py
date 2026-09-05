"""Fixed seed produces fixed output.

Spec §15.
Implemented by DTFM-027.

One of the four test files §13 calls non-negotiable, and the one that underwrites
the others: a physics claim only means something if the run that produced it can
be repeated. §15 puts reproducibility first among the engineering conventions and
§16 requires a reader to "clone it, run one command, and reproduce every figure".

The tests here are stricter than "close enough" in three ways, each closing off a
way a run can be irreproducible while still passing a loose check:

- **byte-identical**, not ``allclose``. A drift of 1e-16 is a different run.
- **across a process boundary**, so hash randomisation, dict ordering and import
  order are exercised rather than assumed away.
- **independent of global random state**, so an unrelated library seeding numpy
  cannot silently change a result.
"""

import hashlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from src import dispersion as dp
from src import generate as gen
from src import noise as nz
from src import tmm_torch as pt

WAVELENGTHS = np.linspace(400.0, 800.0, 64)


def _digest(array: np.ndarray) -> str:
    """Hash of the exact bytes. Two runs agreeing to 1e-16 are different runs."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _run_in_subprocess(source: str) -> str:
    """Execute a snippet in a fresh interpreter and return what it printed.

    A fresh process is the point. Within one process a cached value, a module
    global or an already-seeded generator can make a run look reproducible when
    it is only repeating itself.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


# --- the forward model is deterministic on its own --------------------------


def test_the_forward_model_is_bit_for_bit_repeatable():
    """No seed involved — the simulator must simply be a function."""
    wavelengths = torch.tensor(WAVELENGTHS)
    args = ([420.0, 65.0], [1.0, 1.46, 2.3, 3.88], 0.3, "s")

    first = pt.stack_reflectance(wavelengths, *args).numpy()
    second = pt.stack_reflectance(wavelengths, *args).numpy()

    assert _digest(first) == _digest(second)


def test_the_dispersion_fits_are_repeatable():
    """The nonlinear fits use multi-start. §15 is why the candidate list is a
    fixed sequence rather than random draws — a randomised search would give a
    slightly different answer every run and quietly move every published number.
    """
    assert _digest(np.array(dp.fit_sellmeier("SiO2", oscillators=2).b)) == _digest(
        np.array(dp.fit_sellmeier("SiO2", oscillators=2).b)
    )
    assert dp.fit_cauchy("Si3N4").rms_residual == dp.fit_cauchy("Si3N4").rms_residual
    assert dp.fit_lorentz("Si", oscillators=2).energies == dp.fit_lorentz(
        "Si", oscillators=2
    ).energies


# --- seeded generation ------------------------------------------------------


def test_the_same_seed_gives_byte_identical_data():
    """The acceptance criterion, at its strictest reading."""
    first = gen.generate_batch(WAVELENGTHS, 8, rng=np.random.default_rng(11))
    second = gen.generate_batch(WAVELENGTHS, 8, rng=np.random.default_rng(11))

    assert _digest(first.spectra) == _digest(second.spectra)
    assert _digest(first.targets) == _digest(second.targets)
    assert _digest(first.clean_spectra) == _digest(second.clean_spectra)


def test_a_different_seed_gives_different_data():
    """Guards the test above from passing on a generator that ignores its seed."""
    first = gen.generate_batch(WAVELENGTHS, 8, rng=np.random.default_rng(11))
    other = gen.generate_batch(WAVELENGTHS, 8, rng=np.random.default_rng(12))

    assert _digest(first.spectra) != _digest(other.spectra)


def test_results_do_not_depend_on_global_random_state():
    """The reason every function takes an explicit Generator.

    If the code reached for numpy's global state, an unrelated library seeding it
    — or simply drawing from it — would change results. That is the worst kind of
    irreproducibility: it survives a rerun in the same session and only appears
    when the surrounding code changes.
    """
    np.random.seed(1234)
    first = gen.generate_batch(WAVELENGTHS, 6, rng=np.random.default_rng(5))

    np.random.seed(9999)
    np.random.random(1000)  # disturb the global stream
    torch.manual_seed(4321)
    second = gen.generate_batch(WAVELENGTHS, 6, rng=np.random.default_rng(5))

    assert _digest(first.spectra) == _digest(second.spectra)


def test_the_noise_model_is_seed_reproducible():
    spectrum = pt.stack_reflectance(
        torch.tensor(WAVELENGTHS), [300.0], [1.0, 1.46, 3.88], 0.0, "s"
    ).numpy()

    first = nz.add_detector_noise(spectrum, np.random.default_rng(3))
    second = nz.add_detector_noise(spectrum, np.random.default_rng(3))

    assert _digest(first) == _digest(second)


# --- across a process boundary ----------------------------------------------


def test_two_fresh_processes_agree_exactly():
    """The strongest form of the criterion.

    Within a single process, module-level caches and an already-advanced
    generator can make a run look reproducible when it is only repeating itself.
    Separate interpreters also exercise hash randomisation and import order,
    which PYTHONHASHSEED varies between runs by default.
    """
    source = """
        import hashlib
        import numpy as np
        from src import generate as gen
        wavelengths = np.linspace(400.0, 800.0, 64)
        batch = gen.generate_batch(wavelengths, 6, rng=np.random.default_rng(21))
        print(hashlib.sha256(batch.spectra.tobytes()).hexdigest())
    """
    assert _run_in_subprocess(source) == _run_in_subprocess(source)


def test_the_command_line_entry_point_is_reproducible():
    """§16 requires a reader to clone and reproduce. That promise is about the
    documented commands, not about the library, so it is tested through them.
    """
    def run(seed: int) -> str:
        completed = subprocess.run(
            [sys.executable, "-m", "src.generate", "--config", "configs/default.yaml",
             "--count", "8", "--seed", str(seed)],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    assert run(2) == run(2)
    assert run(2) != run(3)


# --- what reproducibility does not promise ----------------------------------


def test_a_float32_input_is_not_reproducible_against_a_float64_one():
    """Recorded so the difference is not mistaken for irreproducibility.

    Reproducibility is "the same inputs give the same outputs", not "every input
    dtype gives the same answer".

    The subtlety worth pinning: the model promotes everything to complex128, so
    **both calls return float64** and the output dtype tells you nothing. The
    precision lost in the float32 *input* survives that promotion — the values
    differ at ~1e-7, which is float32 resolution, not float64.

    DTFM-015 met the same thing from the other side: a quarter-wave null bottoms
    out at 1e-15 from float32 inputs against 1e-32 from float64. Same
    computation, marginally different question.
    """
    wavelengths_32 = torch.tensor(WAVELENGTHS, dtype=torch.float32)
    wavelengths_64 = torch.tensor(WAVELENGTHS, dtype=torch.float64)
    args = ([1.0, 1.46, 3.88], 0.0, "s")

    from_single = pt.stack_reflectance(
        wavelengths_32, [torch.tensor(420.0, dtype=torch.float32)], *args
    )
    from_double = pt.stack_reflectance(
        wavelengths_64, [torch.tensor(420.0, dtype=torch.float64)], *args
    )

    assert from_single.dtype == torch.float64, "arithmetic is promoted regardless of input"
    assert from_double.dtype == torch.float64

    difference = np.abs(from_single.numpy() - from_double.numpy()).max()
    assert 1e-9 < difference < 1e-5, "float32 resolution, not float64"
    assert _digest(from_single.numpy()) != _digest(from_double.numpy())


@pytest.mark.parametrize("count", [1, 4, 16])
def test_batch_size_does_not_change_the_first_samples(count):
    """A batch of 16 must begin with the same films as a batch of 4 from the
    same seed, or a run's results would depend on how it was chunked.
    """
    reference = gen.sample_parameters(gen.Prior(), 16, np.random.default_rng(8))
    partial = gen.sample_parameters(gen.Prior(), count, np.random.default_rng(8))

    assert np.array_equal(partial["thickness_nm"], reference["thickness_nm"][:count])
