"""The training loop — config-driven, seeded, checkpointed, resumable.

Spec §15.
Implemented by DTFM-043.

§15: "the run that produces a reported number is reproducible from its config."
These pin that property, not the accuracy a run happens to reach.
"""

import json

import numpy as np
import pytest
import torch

from src import training as tr

WAVELENGTHS = np.linspace(400.0, 800.0, 200)


def _tiny(**overrides) -> tr.RunConfig:
    settings = {"width": 32, "depth": 1, "steps": 20, "checkpoint_every": 5}
    return tr.RunConfig(**(settings | overrides))


# --- the config is the record -------------------------------------------------


def test_a_run_writes_the_config_that_produced_it(tmp_path):
    """§15's requirement, made structural. The config lands beside the weights, so
    a result can never be found without the settings that made it.
    """
    tr.train(_tiny(), wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r", progress=False)

    assert (tmp_path / "r" / "config.yaml").exists()
    assert (tmp_path / "r" / "checkpoint.pt").exists()
    assert (tmp_path / "r" / "result.json").exists()

    record = json.loads((tmp_path / "r" / "result.json").read_text())
    for field in ("architecture", "width", "depth", "lr", "seed", "steps"):
        assert field in record, f"{field} missing from the run record"


def test_the_record_says_when_and_where_it_ran(tmp_path):
    """A sweep does not happen in one place or at one time.

    Parallel jobs run on separate machines, so ``inference_us`` and
    ``train_seconds`` are **not** comparable across a sweep — different CPUs, and
    a throttling laptop is slower at the end of a long run than at the start.
    Accuracy is deterministic given the seed and *is* comparable.

    Recording the timestamp and the machine is what lets a reader tell those two
    apart instead of trusting a timing column that has no right to be trusted.
    """
    record = tr.train(_tiny(), wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r",
                      progress=False)

    assert record["when"].endswith("+00:00"), "timestamps are UTC, not local"
    assert record["machine"] and record["commit"]
    assert record["torch"] == torch.__version__


# --- checkpointing and resume -------------------------------------------------


def test_a_killed_run_resumes_instead_of_restarting(tmp_path):
    """The property that makes moving sweeps off a laptop practical.

    Remote jobs get killed — GitHub caps a job at six hours — and a run that
    cannot resume must start from zero. Here a checkpoint truncated to half way
    is picked up and carried to the end.
    """
    config = _tiny(steps=20, checkpoint_every=5)
    tr.train(config, wavelengths_nm=WAVELENGTHS, directory=tmp_path / "a", progress=False)

    saved = tr.Checkpoint.load(tmp_path / "a" / "checkpoint.pt")
    assert saved.step == 20
    saved.step = 10
    saved.save(tmp_path / "b" / "checkpoint.pt")

    tr.train(config, wavelengths_nm=WAVELENGTHS, directory=tmp_path / "b", progress=False)
    assert tr.Checkpoint.load(tmp_path / "b" / "checkpoint.pt").step == 20


def test_resuming_an_already_finished_run_does_not_crash(tmp_path):
    """Re-running a completed sweep entry is ordinary, not an edge case.

    The training loop body never executes, so anything read from a loop variable
    afterwards is unbound. This raised *after* training had finished and the
    weights were on disk — losing the result of a run that had actually succeeded.
    """
    config = _tiny()
    first = tr.train(config, wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r",
                     progress=False)
    again = tr.train(config, wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r",
                     progress=False)

    assert again["median_nm"] == pytest.approx(first["median_nm"], rel=1e-9)
    assert np.isfinite(again["final_loss"])


def test_the_checkpoint_carries_the_optimiser_state(tmp_path):
    """Not optional. Adam holds running moment estimates; resuming without them
    restarts the optimiser cold, which shows as a jump in the loss curve and makes
    a resumed run differ from an uninterrupted one.
    """
    tr.train(_tiny(), wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r", progress=False)
    saved = tr.Checkpoint.load(tmp_path / "r" / "checkpoint.pt")

    assert saved.optimiser["state"], "Adam's moment estimates were not saved"
    assert saved.schedule, "the learning-rate schedule was not saved"
    assert saved.curve, "the loss curve was not saved"


def test_a_run_outside_the_repository_still_records_its_directory(tmp_path):
    """``relative_to`` raises rather than falling back, so a run directory outside
    the repository — a test, or a custom output path — crashed *after* training,
    with the weights already saved.
    """
    record = tr.train(_tiny(), wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r",
                      progress=False)
    assert record["run_dir"]


# --- sweeps -------------------------------------------------------------------


def test_a_sweep_expands_to_the_runs_it_describes():
    runs = tr.expand_sweep({"steps": 100, "sweep": {"depth": [2, 3], "seed": [0, 1]}})

    assert len(runs) == 4
    assert all(run.steps == 100 for run in runs), "unswept fields are inherited"
    assert {(r.depth, r.seed) for r in runs} == {(2, 0), (2, 1), (3, 0), (3, 1)}


def test_a_sweep_drops_combinations_that_differ_only_in_ignored_fields():
    """The bug this test exists for wastes half a sweep silently.

    An MLP does not read ``channels``; a CNN does not read ``width``. Sweeping
    ``[mlp, cnn] × [128, 256] × [16, 32]`` *looks* like 8 runs and is really 4,
    each computed twice. On a laptop that is 50 wasted minutes. On a parallel
    matrix it is four machines producing rows that agree with each other and look
    like reassuring reproducibility.
    """
    runs = tr.expand_sweep({
        "sweep": {"architecture": ["mlp", "cnn"], "width": [128, 256], "channels": [16, 32]}
    })

    assert len(runs) == 4, [r.name for r in runs]
    assert len({r.name for r in runs}) == 4
    assert sum(r.architecture == "mlp" for r in runs) == 2
    assert sum(r.architecture == "cnn" for r in runs) == 2


def test_a_run_name_says_what_the_run_was():
    """The matrix identifies runs by index; the directory has to identify itself."""
    assert tr.RunConfig(architecture="mlp", width=512, depth=4, steps=1000).name == (
        "mlp-w512-d4-s1000-seed0"
    )
    assert tr.RunConfig(architecture="cnn", channels=64, kernel=5, depth=2).name.startswith(
        "cnn-c64k5-d2"
    )


def test_config_without_a_sweep_block_is_a_single_run():
    assert len(tr.expand_sweep({"architecture": "cnn", "steps": 50})) == 1


# --- collecting a parallel sweep ----------------------------------------------


def test_collect_merges_results_and_does_not_duplicate(tmp_path):
    """A parallel sweep returns one artifact per run, so results arrive as a
    directory of files rather than as lines in the log the dashboard reads.

    Idempotence matters because a partial download is normal — a job fails, you
    re-download, and re-collecting must not double every row that came back the
    first time.
    """
    for index in (0, 1):
        directory = tmp_path / f"run-{index}"
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps({"run_dir": f"runs/example-{index}", "median_nm": 1.0 + index})
        )

    history = tmp_path / "history.jsonl"
    assert tr.collect(tmp_path, history=history) == 2
    assert tr.collect(tmp_path, history=history) == 0, "re-collecting must add nothing"
    assert len(history.read_text().strip().splitlines()) == 2


def test_a_run_does_not_write_into_the_project_history(tmp_path):
    """The record the dashboard reads must contain only runs someone meant to make.

    ``train`` used to append to a fixed repository path regardless of where the
    run itself was written, so a test suite — 20 steps, a width-32 network, a
    tmp directory — silently added seven rows to the tuning record. Tuning is
    only a record rather than a memory if nothing else can write to it.
    """
    tr.train(_tiny(), wavelengths_nm=WAVELENGTHS, directory=tmp_path / "runs" / "r",
             progress=False)

    assert (tmp_path / "runs" / "history.jsonl").exists(), "history sits beside the runs"
    assert not (tmp_path / "history.jsonl").exists()
