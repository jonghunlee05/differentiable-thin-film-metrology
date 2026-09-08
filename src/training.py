"""The training loop — config-driven, seeded, checkpointed, resumable.

Spec §15.
Implemented by DTFM-043.

§15's requirement is one sentence and it is the whole design:

    the run that produces a reported number is reproducible from its config

That rules out three things this project had been doing. Hyperparameters on a
command line vanish the moment the shell scrolls. Weights held only in memory die
with the process — DTFM-039 and DTFM-040 produced three trained networks and all
three are gone, leaving only their metrics. And a run that cannot resume must be
restarted from zero when a six-hour job is killed at hour five.

Why resume matters more than it looks
-------------------------------------
The immediate reason is thermal. A CNN run pegs every core for 25 minutes, and a
sweep runs for hours; on a laptop that means throttling, which makes *later runs
in a sweep slower than earlier ones* and quietly corrupts any timing comparison.
The fix is to run sweeps on hardware that is not yours, and remote jobs get
killed — so checkpointing is what makes moving them off the laptop practical.
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import product
from typing import Any

import numpy as np
import torch
import yaml

from src import dataset as ds
from src import generate as gen
from src import losses as ls
from src import models

__all__ = [
    "Checkpoint",
    "RunConfig",
    "collect",
    "evaluate",
    "expand_sweep",
    "load_config",
    "train",
]

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPOSITORY / "runs"
HISTORY = RUNS / "history.jsonl"


@dataclass
class RunConfig:
    """Everything that determines a run's result.

    Deliberately flat and boring. Every field lands in the run record, so the
    record *is* the config — §15's reproducibility requirement is satisfied by
    construction rather than by remembering to write things down.
    """

    architecture: str = "mlp"
    width: int = 256
    depth: int = 3
    channels: int = 32
    kernel: int = 7
    output_margin: float = 0.0
    steps: int = 8000
    batch: int = 256
    lr: float = 1.0e-3
    seed: int = 0
    checkpoint_every: int = 1000
    label: str = ""

    # DTFM-042. `uncertainty` turns on DTFM-041's second head, and is only
    # meaningful together with the NLL it enables: under MSE the head receives no
    # gradient at all, so a run that set this alone would report a sigma frozen at
    # initialisation. `train` refuses that combination rather than producing it.
    uncertainty: bool = False
    #: Weight on the physics reconstruction term. 0 reproduces every run in the
    #: existing 141-run history exactly, which is what makes them comparable to
    #: anything measured after this ticket. DTFM-045 settles the value.
    lambda_recon: float = 0.0

    def model_settings(self) -> dict[str, Any]:
        settings = {
            "architecture": self.architecture,
            "depth": self.depth,
            "output_margin": self.output_margin,
            "width": self.width if self.architecture == "mlp" else 128,
            "uncertainty": self.uncertainty,
        }
        if self.architecture == "cnn":
            settings |= {"channels": self.channels, "kernel": self.kernel}
        return settings

    @property
    def name(self) -> str:
        """A directory name that says what the run was without opening it.

        **Every field that changes the result must appear here**, because
        :func:`expand_sweep` deduplicates on this string. A field left out does
        not merely produce a confusing directory name — it silently collapses a
        sweep. That was DTFM-043's bug: architectures ignoring each other's
        fields turned eight runs into four, and the four missing ones did not
        look missing, they looked like machines reproducing each other.

        ``lambda_recon`` is here for exactly that reason. DTFM-045 is an ablation
        over it, and without it in the name every arm of that ablation would
        deduplicate down to a single run.
        """
        shape = (
            f"w{self.width}"
            if self.architecture == "mlp"
            else f"c{self.channels}k{self.kernel}"
        )
        stem = f"{self.architecture}-{shape}-d{self.depth}-s{self.steps}"
        if self.uncertainty:
            stem += "-nll"
        # Optimiser and output settings appear only when they differ from the
        # default. That keeps the 216 runs already recorded matching a re-run of
        # the same settings — renaming the default would make the whole history
        # incomparable to everything measured afterwards — while still giving a
        # sweep over any of them distinct names. `g` keeps 0.003 as "0.003"
        # rather than "0.003000".
        for prefix, value, default in (
            ("lr", self.lr, 1.0e-3),
            ("b", self.batch, 256),
            ("m", self.output_margin, 0.0),
            ("rec", self.lambda_recon, 0.0),
        ):
            if value != default:
                stem += f"-{prefix}{value:g}"
        stem += f"-seed{self.seed}"
        return f"{self.label}-{stem}" if self.label else stem


@dataclass
class Checkpoint:
    """A run's saved state — weights, optimiser, and where it had got to.

    The optimiser state is saved alongside the weights and is not optional. Adam
    carries running moment estimates; resuming without them restarts the
    optimiser cold, which shows up as a visible jump in the loss curve and makes
    a resumed run differ from an uninterrupted one. That would defeat the point.
    """

    step: int
    model: dict
    optimiser: dict
    schedule: dict
    config: dict
    curve: list = field(default_factory=list)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: pathlib.Path) -> Checkpoint:
        return cls(**torch.load(path, weights_only=False))


def _environment() -> dict[str, str]:
    """Where a run happened, because a sweep does not happen in one place.

    Parallel GitHub jobs run on separate machines, so ``inference_us`` and
    ``train_seconds`` are **not** comparable across a sweep — different CPUs, and
    a throttling laptop is slower at the end of a long run than at the start.
    Accuracy is deterministic given the seed and is comparable; timing is not, and
    recording the machine is what lets a reader tell the two apart.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True, cwd=REPOSITORY,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "commit": commit,
        "machine": f"{platform.system()} {platform.machine()}",
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
    }


def load_config(path: str | pathlib.Path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text()) or {}


def expand_sweep(config: dict) -> list[RunConfig]:
    """Turn one config with a ``sweep`` block into the list of runs it describes.

    A sweep is the Cartesian product of the swept fields, with everything else
    inherited from the base. Each result is a complete :class:`RunConfig` that
    could have been written by hand, so a single run from a sweep is reproducible
    on its own.

    **Duplicates are dropped, and that is not tidiness.** Sweeping across
    architectures produces combinations that differ only in fields the
    architecture ignores — an MLP does not read ``channels``, a CNN does not read
    ``width``. Sweeping ``[mlp, cnn] x [128, 256] x [16, 32]`` looks like 8 runs
    and is really 4, each computed twice. On a laptop that is 50 wasted minutes;
    on a parallel matrix it is 4 machines producing rows that silently agree with
    each other and look like reassuring reproducibility.

    :attr:`RunConfig.name` already encodes only the fields that matter to each
    architecture, so it is the right key to deduplicate on.
    """
    base = {k: v for k, v in config.items() if k != "sweep"}
    sweep = config.get("sweep") or {}
    if not sweep:
        return [RunConfig(**base)]

    keys = list(sweep)
    runs, seen = [], set()
    for values in product(*(sweep[k] for k in keys)):
        candidate = RunConfig(**(base | dict(zip(keys, values, strict=True))))
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        runs.append(candidate)
    return runs


def evaluate(model, prior, wavelengths_nm, films: int = 2000, seed: int = 999) -> dict:
    """Error on unseen films, stratified — §10's protocol.

    DTFM-036 measured why a single number will not do: every method that mostly
    works reports the same median, so one figure cannot separate a working
    estimator from one that fails on a tenth of its films.
    """
    model.eval()
    batch = ds.sample_batch(films, wavelengths_nm, np.random.default_rng(seed), prior=prior)
    with torch.no_grad():
        # `predict` rather than `forward`: it returns sigma in nanometres of
        # thickness, and copes with either head arrangement. Calling the model
        # directly would hand back a tuple once DTFM-041's head is on, and
        # `predicted[:, 0]` would then silently index the first *parameter of the
        # tuple* rather than the thickness.
        predicted, sigma = model.predict(batch.observed.float())
    error = (predicted[:, 0] - batch.targets[:, 0]).numpy()
    truth = batch.targets[:, 0].numpy()

    report = {
        "median_nm": float(np.median(np.abs(error))),
        "rmse_nm": float(np.sqrt(np.mean(error**2))),
        "p95_nm": float(np.percentile(np.abs(error), 95)),
        "wrong_over_1nm": float(np.mean(np.abs(error) > 1.0)),
        "outside_prior": float(model.scale_theta.outside_prior(predicted).float().mean()),
    }
    if sigma is not None:
        # A first, cheap read on whether sigma means anything: the correlation
        # between the uncertainty the model states and the error it actually
        # makes. Calibration proper — coverage, reliability diagrams, the
        # Cramer-Rao comparison — is DTFM-048 through DTFM-050. This is only
        # enough to tell a trained head from an untrained one at a glance.
        sigma_nm = sigma[:, 0].numpy()
        report |= {
            "sigma_median_nm": float(np.median(sigma_nm)),
            "sigma_error_correlation": float(
                np.corrcoef(sigma_nm, np.abs(error))[0, 1]
            ),
            "within_one_sigma": float(np.mean(np.abs(error) <= sigma_nm)),
        }
    for name, low, high in (("thin", 0, 100), ("mid", 100, 700), ("thick", 700, np.inf)):
        mask = (truth >= low) & (truth < high)
        if mask.any():
            report[f"median_{name}"] = float(np.median(np.abs(error[mask])))
    return report


def train(
    config: RunConfig,
    *,
    wavelengths_nm: np.ndarray | None = None,
    directory: pathlib.Path | None = None,
    resume: bool = True,
    progress: bool = True,
    history: pathlib.Path | None = None,
) -> dict:
    """Train one configuration, checkpointing as it goes, and record the result.

    Resumes from ``checkpoint.pt`` if one exists and ``resume`` is set, which is
    what makes a killed six-hour job cost minutes rather than everything.
    """
    wavelengths_nm = (
        np.linspace(400.0, 800.0, 200) if wavelengths_nm is None else np.asarray(wavelengths_nm)
    )
    directory = directory or (RUNS / config.name)
    # The history lives beside the runs it describes, not at a fixed repository
    # path. A run written anywhere else — a test, a custom output directory — was
    # appending to the project's real history, and seven throwaway 20-step runs
    # landed in the record the dashboard reads.
    history = history or (directory.parent / "history.jsonl")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))

    torch.manual_seed(config.seed)
    prior = gen.Prior()
    model = models.build_model(config.model_settings(), prior=prior)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.steps)

    start, curve = 0, []
    checkpoint_path = directory / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        saved = Checkpoint.load(checkpoint_path)
        model.load_state_dict(saved.model)
        optimiser.load_state_dict(saved.optimiser)
        schedule.load_state_dict(saved.schedule)
        start, curve = saved.step, saved.curve
        if progress:
            print(f"  resumed from step {start}", flush=True)

    scaler = model.scale_theta
    rng = np.random.default_rng(config.seed + start)   # not the same films twice
    began, generating = time.perf_counter(), 0.0

    # A resumed run may already be complete — re-running a finished sweep entry
    # is the ordinary case, not an edge one. The loop body then never executes,
    # so `loss` must come from the saved curve rather than from a variable that
    # was never assigned. Left unset this raised *after* training, with the
    # weights already on disk and the result lost.
    final_loss = curve[-1][1] if curve else float("nan")
    breakdown: list[tuple[int, dict[str, float]]] = []

    # Built once per run, not per step: the substrate's nk table and the
    # wavelength grid are fixed by the instrument, and reloading them 120,000
    # times would cost more than the term they serve.
    reconstruct = (
        ls.Reconstructor(wavelengths_nm, prior=prior) if config.lambda_recon else None
    )

    for step in range(start + 1, config.steps + 1):
        mark = time.perf_counter()
        batch = ds.sample_batch(config.batch, wavelengths_nm, rng, prior=prior)
        generating += time.perf_counter() - mark

        model.train()
        optimiser.zero_grad()
        predicted = model(batch.observed.float())
        theta_hat, log_var = predicted if isinstance(predicted, tuple) else (predicted, None)
        loss, parts = ls.total_loss(
            theta_hat,
            batch.targets,
            batch.observed,
            scaler,
            log_var=log_var,
            reconstruct=reconstruct,
            lambda_recon=config.lambda_recon,
        )
        loss.backward()
        optimiser.step()
        schedule.step()
        final_loss = float(loss.item())

        if step % max(config.steps // 100, 1) == 0:
            curve.append((step, float(loss.item())))
            # The parts alongside the total. With two terms in different units a
            # single number cannot say which one moved, and DTFM-045's ablation
            # reads exactly these. Recorded on the same stride as the curve so the
            # two are always the same length.
            breakdown.append((step, parts))
        if step % config.checkpoint_every == 0 or step == config.steps:
            Checkpoint(
                step, model.state_dict(), optimiser.state_dict(),
                schedule.state_dict(), asdict(config), curve,
            ).save(checkpoint_path)
        if progress and step % max(config.steps // 8, 1) == 0:
            print(f"  step {step:6d}   loss {loss.item():.6f}   "
                  f"{time.perf_counter() - began:6.1f}s", flush=True)

    elapsed = time.perf_counter() - began

    # Inference timing: warm up, then average. A single timed call measures
    # dispatch overhead and once reported 11 ms for a 3-layer MLP — a
    # thousandfold error on the network's whole selling point.
    sample = ds.sample_batch(1, wavelengths_nm, np.random.default_rng(1), prior=prior)
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            model(sample.observed.float())
        mark = time.perf_counter()
        for _ in range(500):
            model(sample.observed.float())
        per_film = (time.perf_counter() - mark) / 500

    record = {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        **_environment(),
        **asdict(config),
        "parameters": model.parameter_count,
        **evaluate(model, prior, wavelengths_nm),
        "final_loss": final_loss,
        "loss_curve": curve,
        "loss_parts": breakdown,
        "train_seconds": elapsed,
        "generating_fraction": generating / elapsed,
        "inference_us": per_film * 1e6,
        # Relative when the run lives in the repository, absolute otherwise. A
        # test or a custom output directory can sit anywhere, and `relative_to`
        # raises rather than falling back — which turned a passing run into a
        # crash *after* the training had finished and the weights were saved.
        "run_dir": str(
            directory.relative_to(REPOSITORY)
            if directory.is_relative_to(REPOSITORY)
            else directory
        ),
    }
    (directory / "result.json").write_text(json.dumps(record, indent=2))
    append(record, history)
    return record


def append(record: dict, history: pathlib.Path | None = None) -> None:
    history = history or HISTORY
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def collect(directory: str | pathlib.Path, history: pathlib.Path | None = None) -> int:
    """Merge ``result.json`` files from a downloaded sweep into the history.

    A parallel sweep returns one artifact per run, so the results arrive as a
    directory of files rather than as lines in the log the dashboard reads. This
    is the join, and it is idempotent: a run already in the history — matched on
    its directory name — is skipped, so re-running after a partial download does
    not duplicate rows.
    """
    history = history or HISTORY
    existing = set()
    if history.exists():
        for line in history.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line).get("run_dir"))

    added = 0
    for path in sorted(pathlib.Path(directory).rglob("result.json")):
        record = json.loads(path.read_text())
        if record.get("run_dir") in existing:
            continue
        append(record, history)
        existing.add(record.get("run_dir"))
        added += 1
    return added
