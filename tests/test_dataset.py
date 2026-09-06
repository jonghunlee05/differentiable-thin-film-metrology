"""On-the-fly training data — §7.1's loop.

Spec §7.1.
Implemented by DTFM-038.

These pin the *contract*, not the throughput. The speed number changes with the
machine; what must not change is that the network trains on the same physics the
classical baseline was measured against.
"""

import numpy as np
import pytest
import torch

from src import dataset as ds
from src import generate as gen

WAVELENGTHS = np.linspace(400.0, 800.0, 200)


# --- the contract with the rest of the project --------------------------------


@pytest.mark.parametrize("roughness_range", [(0.0, 0.0), (0.0, 4.0)])
def test_the_vectorised_physics_is_identical_to_the_per_film_loop(roughness_range):
    """The one test this module cannot pass approximately.

    ``sample_batch`` is 19x faster than ``generate.generate_batch`` because it
    runs every film through the transfer matrix at once. If that changed the
    physics even slightly, the network would be trained on a different simulator
    from the one the classical baseline was measured against, and DTFM-044's
    comparison would be meaningless — silently, with no test failing.

    Both roughness settings are checked because roughness is the case most likely
    to break: it adds a Bruggeman layer whose thickness *and* index differ per
    film, so the batched stack has to carry a per-film column rather than a
    scalar.

    **Roughness is varied through the prior, not through** ``Corruption``. An
    earlier version parametrised over ``Corruption(roughness_nm=...)`` and ran
    the identical case twice: both code paths overwrite that field with the
    per-film value drawn from the prior, so the parameter being varied did
    nothing. A parametrised test whose cases are the same is indistinguishable
    from a real one at a glance, which is the point of
    ``test_roughness_actually_changes_the_spectrum`` below — it proves the knob
    being turned here is connected to anything.

    Only the **clean** spectra are compared. The per-film loop consumes the
    random generator in a different order from a vectorised draw, so the noisy
    spectra cannot be bit-identical at the same seed — and need not be, since
    what must match is the physics, not the noise realisation.
    """
    prior = gen.Prior(roughness_nm=roughness_range)
    looped = gen.generate_batch(WAVELENGTHS, 6, rng=np.random.default_rng(3), prior=prior)
    batched = ds.sample_batch(6, WAVELENGTHS, np.random.default_rng(3), prior=prior)

    assert np.array_equal(looped.targets, batched.targets.numpy()), "same films"
    assert np.max(np.abs(looped.clean_spectra - batched.clean.numpy())) == 0.0, (
        "and bit-identical physics, not merely close"
    )


def test_a_batch_has_the_shape_the_network_will_expect():
    batch = ds.sample_batch(8, WAVELENGTHS, np.random.default_rng(0))

    assert batch.observed.shape == (8, 2 * WAVELENGTHS.size)  # Ψ and Δ
    assert batch.targets.shape == (8, 3)  # d, A, B
    assert batch.clean.shape == batch.observed.shape
    assert len(batch) == 8
    assert torch.isfinite(batch.observed).all()
    assert torch.isfinite(batch.targets).all()


def test_the_targets_lie_inside_the_prior():
    """§7.1: "The prior you sample from *is* the model's implicit prior."

    If training data escaped the prior's support, §10's out-of-distribution
    probes would be testing something the network had in fact already seen.
    """
    prior = gen.Prior()
    batch = ds.sample_batch(64, WAVELENGTHS, np.random.default_rng(1), prior=prior)
    d, a, b = batch.targets.T.numpy()

    assert np.all((prior.thickness_nm[0] <= d) & (d <= prior.thickness_nm[1]))
    assert np.all((prior.cauchy_a[0] <= a) & (a <= prior.cauchy_a[1]))
    assert np.all((prior.cauchy_b[0] <= b) & (b <= prior.cauchy_b[1]))


# --- properties of an endless stream ------------------------------------------


def test_the_network_never_sees_the_same_film_twice():
    """The reason §7.1 generates rather than stores. There is no fixed sample to
    overfit and no train/validation split to draw, because every batch is new.
    """
    stream = ds.FilmStream(batch_size=16, wavelengths_nm=WAVELENGTHS, batches=3, seed=0)
    seen = [b.targets[:, 0].numpy() for b in stream]

    assert len(seen) == 3
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not np.any(np.isin(seen[i], seen[j])), "batches must not repeat films"


def test_an_infinite_stream_still_replays_from_its_seed():
    """§15. Infinite is not an excuse: a training run that cannot be repeated
    makes every number it produces unfalsifiable.
    """
    first = [
        b.targets.numpy()
        for b in ds.FilmStream(batch_size=8, wavelengths_nm=WAVELENGTHS, batches=2, seed=7)
    ]
    again = [
        b.targets.numpy()
        for b in ds.FilmStream(batch_size=8, wavelengths_nm=WAVELENGTHS, batches=2, seed=7)
    ]
    other = [
        b.targets.numpy()
        for b in ds.FilmStream(batch_size=8, wavelengths_nm=WAVELENGTHS, batches=2, seed=8)
    ]

    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not np.array_equal(first[0], other[0]), "a different seed gives different films"


def test_each_worker_draws_a_different_stream():
    """The silent bug this test exists for.

    PyTorch forks the dataset into each worker, and a forked copy carries the
    *same* generator state — so without per-worker seeding every worker produces
    byte-identical films and the effective batch size collapses to one worker's
    worth. Nothing raises. Nothing slows down. It shows up only as a network that
    will not train, weeks later.

    ``SeedSequence.spawn`` gives each worker an independent, non-overlapping
    stream. A test that only checked "batches come out" would pass with the bug
    present, which is why this one checks the films actually differ.
    """
    streams = []
    for worker in (0, 1, 2):
        rng = np.random.default_rng(np.random.SeedSequence(0).spawn(3)[worker])
        streams.append(ds.sample_batch(8, WAVELENGTHS, rng).targets[:, 0].numpy())

    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.any(np.isin(streams[i], streams[j])), (
                f"workers {i} and {j} generated the same films — the batch has collapsed"
            )


def test_the_stream_is_endless_unless_told_otherwise():
    stream = ds.FilmStream(batch_size=4, wavelengths_nm=WAVELENGTHS, seed=0)
    iterator = iter(stream)
    assert all(len(next(iterator)) == 4 for _ in range(5))


# --- the AC's throughput ------------------------------------------------------


def test_throughput_is_measured_and_beats_the_per_film_loop():
    """§7.1's AC asks for throughput, and DTFM-042 makes it load-bearing: that
    ticket runs the simulator *inside* the training step, at which point the
    physics is ~98% of all training cost and the network is ~2%.

    Measured on the development machine: the per-film loop managed 440 films/sec
    against a training step of roughly 2 ms — a hundredfold mismatch that starves
    the network. The vectorised path reached 8,169 films/sec with 4 workers, and
    notably *fell* to 5,820 at 6 workers, because torch already uses 4 threads
    and the two oversubscribe an 8-core machine. More workers is not monotonically
    better, which is why this is measured rather than assumed.

    The assertion is deliberately loose — CI hardware is not this machine — and
    checks only that the vectorised path is comfortably past what the loop did.
    """
    result = ds.throughput(batch_size=64, batches=3, num_workers=0, wavelengths_nm=WAVELENGTHS)

    assert result["films_per_second"] > 1000.0, (
        f"only {result['films_per_second']:.0f} films/sec — slower than expected; "
        "the per-film loop this replaces managed 440"
    )
    assert result["seconds_per_batch"] > 0.0
    assert result["batch_size"] == 64.0


def test_no_generated_data_is_written_anywhere(tmp_path, monkeypatch):
    """§7.1's AC: "no generated data committed". Stronger than a .gitignore rule —
    the pipeline must not have a path that writes a dataset at all.
    """
    monkeypatch.chdir(tmp_path)
    for _ in ds.FilmStream(batch_size=8, wavelengths_nm=WAVELENGTHS, batches=2, seed=0):
        pass

    assert list(tmp_path.iterdir()) == [], "the pipeline wrote files it should not have"


def test_roughness_actually_changes_the_spectrum():
    """The control for the parametrisation above.

    A test that varies a knob is worthless if the knob is disconnected. This one
    holds the films fixed and turns roughness on, then asserts the physics moved.
    Measured: the same six films differ by 6.16 in Ψ/Δ between a smooth prior and
    a rough one, and the sampled heights span 0.01 to 3.89 nm.

    Written after discovering the original parametrisation ran the same case
    twice — see ``Implementation-Notes.md`` §27.
    """
    smooth = gen.generate_batch(
        WAVELENGTHS, 6, rng=np.random.default_rng(3), prior=gen.Prior(roughness_nm=(0.0, 0.0))
    )
    rough = gen.generate_batch(
        WAVELENGTHS, 6, rng=np.random.default_rng(3), prior=gen.Prior(roughness_nm=(0.0, 4.0))
    )

    assert np.allclose(smooth.targets[:, 0], rough.targets[:, 0]), "the same films"
    assert np.all(smooth.roughness_nm == 0.0)
    assert rough.roughness_nm.max() > 1.0, "and roughness was genuinely drawn"
    assert np.max(np.abs(smooth.clean_spectra - rough.clean_spectra)) > 1.0, (
        "roughness must change the spectrum, or the parametrised test above is inert"
    )


def test_a_one_picometre_change_in_thickness_is_visible():
    """Proves the bit-identical comparison is not vacuous.

    ``test_the_vectorised_physics_is_identical_to_the_per_film_loop`` asserts a
    difference of exactly zero, which is only meaningful if the comparison could
    detect a difference at all. It can: perturbing one film by **0.001 nm** — a
    thousandth of a nanometre, about a hundredth of an atom — moves the spectrum
    by 2.3e-05, far above the zero the test demands.
    """
    rng = np.random.default_rng(3)
    sampled = gen.sample_parameters(gen.Prior(), 4, rng)
    nudged = {k: v.copy() for k, v in sampled.items()}
    nudged["thickness_nm"][0] += 0.001

    substrate = ds._substrate(gen.Prior(), WAVELENGTHS)
    both = []
    for params in (sampled, nudged):
        monkey = lambda p, n, r, _v=params: _v  # noqa: E731 - fixed draw, not a fixture
        original, gen.sample_parameters = gen.sample_parameters, monkey
        try:
            both.append(
                ds.sample_batch(4, WAVELENGTHS, np.random.default_rng(0), substrate=substrate)
            )
        finally:
            gen.sample_parameters = original

    moved = np.max(np.abs(both[0].clean.numpy() - both[1].clean.numpy()))
    assert moved > 1e-6, f"a 0.001 nm change moved the spectrum by only {moved:.1e}"
