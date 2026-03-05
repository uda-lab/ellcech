import importlib

import pytest

import ellphi_alpha.backends.gudhi as gudhi_backend_mod
from ellphi_alpha.filtration import FiltrationEntry
from ellphi_alpha.gudhi_bridge import to_gudhi_simplex_tree


def test_to_gudhi_simplex_tree_missing_gudhi(monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "gudhi":
            raise ModuleNotFoundError("No module named 'gudhi'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(gudhi_backend_mod.importlib, "import_module", fake_import_module)
    with pytest.raises(RuntimeError, match="gudhi is not installed"):
        to_gudhi_simplex_tree([((0,), 0.0)])


def test_to_gudhi_simplex_tree_inserts_entries(monkeypatch):
    class FakeSimplexTree:
        def __init__(self):
            self.inserted = []
            self.non_decreasing_called = False

        def insert(self, simplex, filtration):
            self.inserted.append((tuple(simplex), float(filtration)))
            return True

        def make_filtration_non_decreasing(self):
            self.non_decreasing_called = True

    class FakeGudhiModule:
        SimplexTree = FakeSimplexTree

    original_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "gudhi":
            return FakeGudhiModule
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(gudhi_backend_mod.importlib, "import_module", fake_import_module)

    filtration = [
        FiltrationEntry(simplex=(0,), alpha=0.0, predicates=None),
        ((0, 1), 1.5),
    ]
    st = to_gudhi_simplex_tree(filtration)
    assert st.inserted == [((0,), 0.0), ((0, 1), 1.5)]
    assert st.non_decreasing_called
