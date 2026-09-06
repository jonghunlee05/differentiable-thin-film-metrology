"""On-the-fly training data — §7.1's loop, fast enough to keep a network fed.

Spec §7.1.
Implemented by DTFM-038.

§7.1 is unambiguous about where training data comes from:

    The simulator is the labeller, so training data is unlimited and free.
    Generate on the fly rather than storing a dataset.

That removes two failure modes before they can happen. There is no stored dataset
to drift out of step with the physics that produced it, and the network cannot
overfit a fixed sample because **it never sees the same film twice**. It also
makes a train/validation split meaningless — every batch is already unseen.

Why this module exists rather than calling ``generate.generate_batch``
---------------------------------------------------------------------
``generate_batch`` builds films **one at a time in a Python loop**, which was the
right shape for producing a few dozen test cases and is the wrong shape for
producing millions. Measured on this machine:

    per-film loop      440 films/sec
    batched call    16,908 films/sec        38x

The simulator has been able to do this since DTFM-013 ("batch over wavelength and
parameters"); ``generate.py`` simply never used it. The capability was there and
unspent.

That 38x is not a convenience. DTFM-042 adds a reconstruction loss that runs the
simulator *inside* the training step and backpropagates through it, at which
point the physics is **98% of all training cost** and the network is 2%. Every
remaining ticket in the project pays this bill, so it is worth paying once.

What is deliberately not here
-----------------------------
No normalisation, no augmentation, no train/validation split. Normalisation is an
architecture decision and belongs with the model in DTFM-039; a split is
meaningless on an infinite stream. This module's whole job is to produce
``(observation, truth)`` pairs identical to the ones the classical baseline was
measured against — because if training and benchmarking disagree about the
physics, DTFM-044's comparison means nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

from src import dispersion as dp
from src import generate as gen
from src import noise as nz
from src import tmm_torch as pt

__all__ = [
    "FilmStream",
    "TrainingBatch",
    "make_loader",
    "sample_batch",
    "throughput",
]


@dataclass
class TrainingBatch:
    """One batch: what the instrument saw, and what produced it.

    ``clean`` is kept alongside ``observed`` and is **not** a training input —
    a network given the noiseless spectrum would be solving a different, easier
    problem. It is here because DTFM-045's ablation needs to separate "the fit
    failed" from "the noise made it impossible", which requires both.
    """

    observed: torch.Tensor
    targets: torch.Tensor
    clean: torch.Tensor
    wavelengths_nm: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.observed.shape[0])


def _substrate(prior: gen.Prior, wavelengths_nm: NDArray) -> torch.Tensor:
    n, k = dp.load_nk(prior.substrate, wavelengths_nm)
    return torch.tensor(n + 1j * k)


def sample_batch(
    count: int,
    wavelengths_nm: NDArray,
    rng: np.random.Generator,
    *,
    prior: gen.Prior | None = None,
    measurement: gen.Measurement | None = None,
    corruption: nz.Corruption | None = None,
    substrate: torch.Tensor | None = None,
) -> TrainingBatch:
    """``count`` films in one vectorised forward pass.

    The stack is built with a leading dimension over films, so ``count``
    thicknesses and ``count`` dispersion curves go through the transfer matrix
    together rather than in sequence.

    **Surface roughness varies per film and survives that.** §4.5 draws a
    roughness height for each film, and a rough surface is modelled as an extra
    Bruggeman layer whose thickness *is* that height and whose index depends on
    the film's own index. Both therefore differ per film — but the stack's
    *structure* does not, so the layer thicknesses become a ``(count, 1)`` column
    and its indices a ``(count, wavelengths)`` block. The physics is unchanged;
    only the loop is gone.

    The one thing that cannot be shared is the random draw. The per-film loop
    consumes ``rng`` in a different order from a vectorised draw, so noisy
    spectra are *not* bit-identical to ``generate.generate_batch`` at the same
    seed. The **clean** spectra are, and ``test_dataset.py`` asserts exactly
    that: the physics must match, the noise need only have the right
    distribution.
    """
    prior = prior or gen.Prior()
    measurement = measurement or gen.Measurement()
    corruption = corruption or nz.Corruption()
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if substrate is None:
        substrate = _substrate(prior, wavelengths_nm)

    sampled = gen.sample_parameters(prior, count, rng)
    thickness = torch.as_tensor(sampled["thickness_nm"])[:, None]
    cauchy_a = torch.as_tensor(sampled["cauchy_a"])[:, None]
    cauchy_b = torch.as_tensor(sampled["cauchy_b"])[:, None]
    roughness = torch.as_tensor(sampled["roughness_nm"])[:, None]

    grid = torch.as_tensor(wavelengths_nm)[None, :]
    film_index = dp.cauchy_n((cauchy_a, cauchy_b, torch.zeros(())), grid)

    # Roughness: a Bruggeman mixture of film and void, one layer per film.
    layers, media = [thickness], [torch.ones(()), film_index, substrate[None, :]]
    if float(roughness.max()) > 0.0:
        mixed = nz.effective_medium_index(film_index, torch.ones_like(film_index), 0.5)
        layers = [roughness, thickness]
        media = [torch.ones(()), mixed, film_index, substrate[None, :]]

    angle = measurement.angle_rad
    with torch.no_grad():
        if measurement.observable == "ellipsometry":
            psi, delta = pt.stack_psi_delta(grid, layers, media, angle)
            clean = torch.cat([psi, delta], dim=1)

            ratio = pt.stack_rho(grid, layers, media, angle)
            if corruption.bandwidth_fwhm_nm:
                # Smoothing the complex ratio, not Ψ and Δ. Averaging Δ across
                # its ±π wrap invents values no film produces — see generate.py.
                kernel = _slit(wavelengths_nm, corruption.bandwidth_fwhm_nm)
                ratio = torch.complex(ratio.real @ kernel.T, ratio.imag @ kernel.T)
            noise = torch.as_tensor(
                rng.normal(0.0, measurement.ellipsometer_sigma_rad, (count, 2 * grid.shape[1]))
            )
            observed = torch.cat([torch.atan(torch.abs(ratio)), torch.angle(ratio)], dim=1) + noise
        else:
            clean = pt.stack_reflectance(grid, layers, media, angle, "s")
            observed = clean.clone()
            if corruption.bandwidth_fwhm_nm:
                observed = observed @ _slit(wavelengths_nm, corruption.bandwidth_fwhm_nm).T
            observed = observed * corruption.baseline_gain + corruption.baseline_offset
            scale = corruption.shot_scale * torch.sqrt(torch.clamp(observed, min=0.0))
            observed = observed + torch.as_tensor(rng.normal(0.0, 1.0, tuple(observed.shape))) * (
                scale + corruption.read_sigma
            )

    targets = torch.stack([thickness[:, 0], cauchy_a[:, 0], cauchy_b[:, 0]], dim=1)
    return TrainingBatch(observed, targets, clean, wavelengths_nm)


def _slit(wavelengths_nm: NDArray, fwhm_nm: float) -> torch.Tensor:
    """The spectrometer's slit function as a matrix, matching ``noise.py``."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = np.exp(-0.5 * ((wavelengths_nm[:, None] - wavelengths_nm[None, :]) / sigma) ** 2)
    return torch.as_tensor(kernel / kernel.sum(axis=1, keepdims=True))


class FilmStream(IterableDataset):
    """An endless stream of freshly simulated films.

    **Worker seeding is the subtle part.** PyTorch forks this object into each
    worker process, and a forked copy carries the *same* generator state — so
    every worker would produce byte-identical films and the effective batch size
    would silently collapse to one worker's worth. The bug does not raise, does
    not slow anything down, and shows up only as a network that will not train.

    Each worker therefore derives its own stream from ``seed`` and its worker id
    via ``SeedSequence.spawn``, which guarantees independent, non-overlapping
    sequences. ``test_dataset.py`` asserts workers disagree, because a test that
    only checked "it produces batches" would pass with the bug present.
    """

    def __init__(
        self,
        batch_size: int = 256,
        wavelengths_nm: NDArray | None = None,
        *,
        seed: int = 0,
        prior: gen.Prior | None = None,
        measurement: gen.Measurement | None = None,
        corruption: nz.Corruption | None = None,
        batches: int | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.wavelengths_nm = (
            np.linspace(400.0, 800.0, 200)
            if wavelengths_nm is None
            else np.asarray(wavelengths_nm, dtype=float)
        )
        self.seed = int(seed)
        self.prior = prior or gen.Prior()
        self.measurement = measurement or gen.Measurement()
        self.corruption = corruption or nz.Corruption()
        self.batches = batches

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        workers = 1 if info is None else info.num_workers
        rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(workers)[worker])

        substrate = _substrate(self.prior, self.wavelengths_nm)  # loaded once per worker
        produced = 0
        while self.batches is None or produced < self.batches:
            yield sample_batch(
                self.batch_size,
                self.wavelengths_nm,
                rng,
                prior=self.prior,
                measurement=self.measurement,
                corruption=self.corruption,
                substrate=substrate,
            )
            produced += 1


def make_loader(stream: FilmStream, num_workers: int = 0):
    """A ``DataLoader`` over ``stream``, batching already done upstream.

    ``batch_size=None`` because :class:`FilmStream` yields whole batches — the
    simulator is far more efficient called once for 256 films than 256 times for
    one, which is the entire point of this module. Letting the loader collate
    single films would reintroduce the loop it exists to remove.
    """
    return torch.utils.data.DataLoader(
        stream,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def throughput(
    batch_size: int = 256, batches: int = 8, num_workers: int = 0, **stream_options
) -> dict[str, float]:
    """Films per second, which §7.1's AC asks for and DTFM-042 makes load-bearing.

    Worth measuring rather than assuming: if generation is slower than a training
    step, the network starves and the bottleneck is the simulator rather than the
    learning. On this machine the per-film loop gave 440 films/sec against a
    training step of roughly 2 ms — a hundredfold mismatch that the vectorised
    path removes.

    The first batch is discarded. It carries one-off import and allocation cost
    that would otherwise be charged to the steady-state rate.
    """
    stream = FilmStream(batch_size=batch_size, batches=batches + 1, **stream_options)
    loader = make_loader(stream, num_workers=num_workers)

    started = None
    counted = 0
    for index, batch in enumerate(loader):
        if index == 0:
            started = time.perf_counter()  # discard the warm-up batch
            continue
        counted += len(batch)
    elapsed = time.perf_counter() - started

    return {
        "films_per_second": counted / elapsed,
        "seconds_per_batch": elapsed / max(batches, 1),
        "batch_size": float(batch_size),
        "num_workers": float(num_workers),
    }
