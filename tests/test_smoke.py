"""Package smoke tests: the tree imports, and `tmm` stays out of src/.

Spec §13, §15.
Implemented by DTFM-004.
"""

import importlib
import pathlib
import pkgutil

import nbformat
import pytest

import src

SRC_MODULES = [f"src.{m.name}" for m in pkgutil.iter_modules(src.__path__)]


def test_module_list_is_not_empty():
    assert SRC_MODULES, "no modules found under src/ — the package tree is missing"


@pytest.mark.parametrize("name", SRC_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", SRC_MODULES)
def test_src_never_imports_tmm(name):
    """Standing decision, spec §0.1: write the forward model; import `tmm` only
    to validate. `tmm` is a dev-only dependency, so a src/ module reaching for it
    would break a clean runtime install — catch that here rather than in the field.
    """
    source = importlib.import_module(name).__loader__.get_source(name) or ""
    offenders = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import tmm", "from tmm"))
    ]
    assert not offenders, f"{name} imports the reference package: {offenders}"


# --- notebooks are deliverables, so a broken one must not look finished -------

NOTEBOOKS = sorted((pathlib.Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_cell_was_actually_executed(path):
    """§13 ships notebooks as narrative. An unexecuted one is a claim, not a
    demonstration — and a *partly* executed one is worse, because it looks
    finished while a cell quietly did nothing.

    Checked on ``execution_count`` rather than on the presence of output. An
    earlier version of this test asserted every code cell produced output and
    failed immediately on notebook 01, whose import cell legitimately prints
    nothing. ``execution_count`` is ``None`` until a cell runs, which is the
    property actually being claimed.

    Checked here rather than left to review: notebooks are build artefacts of
    ``scripts/build_notebook_*.py``, so it is easy to rebuild one — which clears
    every output — and commit it without re-running.
    """
    notebook = nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert code, f"{path.name} has no code cells"
    unrun = [i for i, cell in enumerate(code) if cell.get("execution_count") is None]
    assert not unrun, f"{path.name}: code cells {unrun} were never executed"
    assert any(cell.outputs for cell in code), f"{path.name} produced no output at all"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_the_notebook_contains_no_errors(path):
    """A traceback in a committed notebook is a failing test nobody is running."""
    notebook = nbformat.read(path, as_version=4)
    errors = [
        (i, output.get("ename"))
        for i, cell in enumerate(notebook.cells)
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors, f"{path.name} contains error outputs: {errors}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_has_a_build_script(path):
    """§15: "logic lives in src/, narrative in notebooks/" — and a notebook is a
    JSON blob whose diffs are unreadable, so the authored form is plain python
    and the .ipynb is a build artefact. A notebook with no build script has no
    reviewable source.
    """
    number = path.name.split("_")[0]
    script = path.parent.parent / "scripts" / f"build_notebook_{number}.py"
    assert script.exists(), f"{path.name} has no {script.name}"
