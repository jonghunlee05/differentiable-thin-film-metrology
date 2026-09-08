"""The training loop — config-driven, seeded, checkpointed, resumable.

Spec §15.
Implemented by DTFM-043.

§15: "the run that produces a reported number is reproducible from its config."
These pin that property, not the accuracy a run happens to reach.
"""

import dataclasses
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


# --- DTFM-042: the losses reaching the training loop --------------------------


def test_the_default_run_name_is_unchanged_by_the_new_fields():
    """141 runs are already in the history under the old naming.

    If adding the DTFM-042 fields renamed the default configuration, every one of
    those rows would stop matching a re-run of the same settings, and the whole
    recorded history would become incomparable to anything measured afterwards.
    """
    assert tr.RunConfig().name == "mlp-w256-d3-s8000-seed0"


def test_the_run_name_distinguishes_the_loss_settings():
    """``expand_sweep`` deduplicates on the name, so a field missing from it does
    not produce a confusing directory — it silently deletes runs.

    DTFM-045 ablates λ_recon. Without it in the name every arm of that ablation
    collapses to one run, and the collapse looks like agreement rather than loss.
    """
    assert tr.RunConfig(uncertainty=True).name == "mlp-w256-d3-s8000-nll-seed0"
    assert tr.RunConfig(uncertainty=True, lambda_recon=0.1).name == (
        "mlp-w256-d3-s8000-nll-rec0.1-seed0"
    )


def test_an_ablation_over_lambda_recon_does_not_deduplicate():
    runs = tr.expand_sweep(
        {"uncertainty": True, "sweep": {"lambda_recon": [0.0, 0.01, 0.1, 1.0]}}
    )

    assert len({r.name for r in runs}) == 4, [r.name for r in runs]


def test_a_run_with_the_head_records_what_its_uncertainty_did(tmp_path):
    """Not calibration — that is DTFM-048 through DTFM-050. Enough to tell a
    trained head from an untrained one without opening the checkpoint.
    """
    record = tr.train(
        _tiny(uncertainty=True, lambda_recon=0.1),
        wavelengths_nm=WAVELENGTHS,
        directory=tmp_path / "r",
        progress=False,
    )

    for field in ("sigma_median_nm", "sigma_error_correlation", "within_one_sigma"):
        assert field in record, f"{field} missing"
    assert np.isfinite(record["sigma_median_nm"])
    assert record["sigma_median_nm"] > 0


def test_a_run_without_the_head_reports_no_uncertainty_fields(tmp_path):
    record = tr.train(
        _tiny(), wavelengths_nm=WAVELENGTHS, directory=tmp_path / "r", progress=False
    )

    assert "sigma_median_nm" not in record


def test_the_loss_parts_are_recorded_alongside_the_total(tmp_path):
    """With two terms in different units the total cannot say which one moved.

    DTFM-045 reads these to attribute a change to the reconstruction term rather
    than to the fit, and a run log holding only the total makes that impossible
    after the fact rather than merely inconvenient.
    """
    record = tr.train(
        _tiny(uncertainty=True, lambda_recon=0.5),
        wavelengths_nm=WAVELENGTHS,
        directory=tmp_path / "r",
        progress=False,
    )

    assert record["loss_parts"], "no breakdown recorded"
    _, parts = record["loss_parts"][-1]
    assert set(parts) == {"fit", "recon", "total"}
    assert parts["total"] == pytest.approx(parts["fit"] + 0.5 * parts["recon"], rel=1e-5)


def test_the_physics_term_is_skipped_entirely_when_its_weight_is_zero(tmp_path):
    """λ_recon = 0 must not run the transfer matrix and multiply it by nothing.

    The reconstruction term is the expensive part of a step — a full stack over
    200 wavelengths per film — so computing and discarding it would make the
    control arm of DTFM-045's ablation as slow as its treatment arm, and every
    run in the existing history slower for no result.
    """
    record = tr.train(
        _tiny(uncertainty=True, lambda_recon=0.0),
        wavelengths_nm=WAVELENGTHS,
        directory=tmp_path / "r",
        progress=False,
    )

    _, parts = record["loss_parts"][-1]
    assert "recon" not in parts


# --- run naming, guarded as a property rather than case by case ---------------


@pytest.mark.parametrize(
    ("architecture", "field", "value"),
    [
        ("mlp", "width", 128),
        ("mlp", "depth", 2),
        ("mlp", "steps", 100),
        ("mlp", "seed", 1),
        ("mlp", "lr", 3.0e-3),
        ("mlp", "batch", 64),
        ("mlp", "output_margin", 0.2),
        ("mlp", "uncertainty", True),
        ("mlp", "lambda_recon", 1.0e-3),
        ("cnn", "channels", 16),
        ("cnn", "kernel", 5),
        ("cnn", "depth", 2),
        ("cnn", "lr", 3.0e-3),
        ("cnn", "batch", 64),
    ],
)
def test_every_field_that_changes_a_result_changes_the_run_name(architecture, field, value):
    """The one property that keeps a sweep honest.

    ``expand_sweep`` deduplicates on the name, so a result-affecting field
    missing from it does not produce a confusing directory — it silently deletes
    runs, and the deletion looks like agreement rather than loss.

    This has now happened twice. DTFM-043: architectures ignoring each other's
    fields turned eight runs into four. DTFM-042: ``lambda_recon`` was missing, so
    an ablation over it would have collapsed to one run. Both were found by
    accident. Enumerating the fields here turns the next one into a failing test
    instead of a wasted night — ``lr``, ``batch`` and ``output_margin`` were all
    missing when this was written, and a sweep over learning rate would have run
    a third of the jobs it claimed to.

    Add a row here whenever ``RunConfig`` gains a field that changes a result.
    """
    base = tr.RunConfig(architecture=architecture)
    changed = dataclasses.replace(base, **{field: value})

    assert changed.name != base.name, f"{field} does not appear in the run name"


@pytest.mark.parametrize("field", ["checkpoint_every"])
def test_fields_that_do_not_change_a_result_stay_out_of_the_name(field):
    """The converse, so the rule above is not satisfied by naming everything.

    How often a checkpoint is written changes what is on disk during a run, not
    what the run produces. Putting it in the name would make two identical runs
    look different and defeat the deduplication.
    """
    base = tr.RunConfig()

    assert dataclasses.replace(base, **{field: 7}).name == base.name


def test_an_architecture_ignoring_a_field_still_deduplicates():
    """The behaviour that must survive the fix above.

    An MLP does not read ``channels``. Two MLP runs differing only in it are the
    same network trained twice from the same seed, and collapsing them is correct
    — that is DTFM-043's dedup doing its job, not a naming gap.
    """
    assert tr.RunConfig(architecture="mlp", channels=16).name == (
        tr.RunConfig(architecture="mlp", channels=64).name
    )
