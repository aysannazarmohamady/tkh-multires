"""Guards against the embedding scripts drifting from src/method.py.

Run: python -m pytest tests/  (or: python tests/test_filter_consistency.py)
Heavy deps (sentence_transformers, gradio) are stubbed; only filtering is tested.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for mod in ("sentence_transformers", "gradio", "spaces"):
    stub = types.ModuleType(mod)
    stub.SentenceTransformer = object
    stub.Interface = lambda *a, **k: None
    stub.File = stub.Textbox = lambda *a, **k: None
    sys.modules.setdefault(mod, stub)
del sys.modules["spaces"]  # app.py falls back to a no-op decorator

import method  # noqa: E402

DATA = json.load(open(ROOT / "data/tkh_collection10.json"))
CUTOFFS = [2020, 2022, 2024, 2026]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SCRIPTS = [_load("compute_embeddings_hf.py", "ce_hf"), _load("hf_space/app.py", "hf_app")]


def test_exclusion_sets_match():
    for s in SCRIPTS:
        assert s.EXCLUDED_FROM_CLUSTERING == method.EXCLUDED_FROM_CLUSTERING
        assert s.HELD_OUT_FOR_COHERENCE == method.HELD_OUT_FOR_COHERENCE


def test_edge_sets_and_order_match_every_cutoff():
    for c in CUTOFFS:
        ref = [e["id"] for e in method.filter_snapshot(DATA, c)[1]]
        for s in SCRIPTS:
            assert [e["id"] for e in s.filter_clustering_edges(DATA, c)] == ref, (s.__name__, c)


def test_shipped_embedding_order_is_2026_edge_order():
    shipped = json.load(open(ROOT / "outputs/embedding_edge_order.json"))
    assert shipped == [e["id"] for e in method.filter_snapshot(DATA, 2026)[1]]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
