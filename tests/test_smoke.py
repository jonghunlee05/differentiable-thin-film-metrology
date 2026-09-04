"""Package smoke tests: the tree imports, and `tmm` stays out of src/.

Spec §13, §15.
Implemented by DTFM-004.
"""

import importlib
import pkgutil

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
