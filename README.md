# tkh-multires

Multi-resolution semantic abstraction over a temporal knowledge hypergraph —
hyperedge-aware hierarchical clustering with temporal stability tracking.

## Status

Work in progress.

## Task checklist

- [x] T1 — Load and describe the evolving graph
- [x] Literature review (prep for T2) — see `literature_review.md`
- [ ] T2 — Method and formal statement (Deliverable 0)
  - [x] Deliverable 0 draft (formal problem statement) — see `report.md`
  - [x] Self-review of Deliverable 0 against real data; design fixed
        (similarity formula, node-assignment note, coherence-circularity
        design decision) — see `AI_USAGE.md`
  - [x] Method implementation (hyperedge clustering + node assignment) —
        see `src/method.py`; verified on the full 2026 snapshot: 702
        clustering edges, level-0 cluster count 13-14 (target 10-15),
        laminar check PASS, 100% node coverage after post-hoc attachment
- [ ] T3 — Temporal coupling
- [ ] T4 — Hyper-edge collapse
- [ ] T5 — Labelling with measured faithfulness
- [ ] T6 — Evaluation
- [ ] T7 — Write-up

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Using **Option A**: a provided temporal TKH export, covering 52 articles,
~5,800 nodes, ~1,400 hyper-edges (years 1901-2026, with most activity
concentrated in recent years).

- `data/tkh_collection10.json` — the graph itself.
- `data/questions.csv`, `data/ground_truth.json` — reserved for the T6
  extrinsic, task-grounded evaluation (coarse-to-fine retrieval hit-rate /
  steps-to-target), not yet used.

## Semantic embeddings (Hugging Face Space)

`src/method.py`'s sandbox has no network access to huggingface.co, so it
falls back to offline TF-IDF for the semantic signal by default. For real
sentence-transformer embeddings (recommended — see `report.md` for why:
TF-IDF correlates too strongly with the structural signal, ~0.69, while real
embeddings bring it down to ~0.31), we compute them separately on a machine
with internet access:

- `hf_space/` contains a small Gradio app (deployed as a Hugging Face
  Space) that takes `data/tkh_collection10.json` as input and outputs
  `semantic_embeddings.npy` + `embedding_edge_order.json`, using the same
  hyperedge filtering as `src/method.py` (excludes `claims`, holds out
  `cites`).
- Live Space: **https://huggingface.co/spaces/aysan98/tkh**
- After downloading the two output files into `outputs/`, run:
  ```bash
  python src/method.py --cutoff 2026 \
      --embeddings-path outputs/semantic_embeddings.npy \
      --embeddings-order outputs/embedding_edge_order.json
  ```
- `compute_embeddings_hf.py` (repo root) is the equivalent as a plain script,
  for running on any internet-connected machine instead of the Space.

## Reproduction

```bash
python src/load_graph.py --data data/tkh_collection10.json
```

Produces per-snapshot descriptive stats (node/edge counts, node-type
distribution, hyper-edge arity distribution, degree distribution and
singleton fraction, growth between snapshots by type, duplicate
surface-form detection, and data-quality notes) in
`outputs/t1_snapshot_stats.json`.

```bash
python src/method.py --cutoff 2026 \
    --embeddings-path outputs/semantic_embeddings.npy \
    --embeddings-order outputs/embedding_edge_order.json
```

Runs the core T2 method (see `report.md` for the full design): builds a
combined structural+semantic similarity between hyperedges, runs
hierarchical agglomerative clustering, extracts a level-0 cut targeting
10-15 clusters (P2), assigns nodes top-down so the result is laminar by
construction (P1, empirically verified each run), and attaches leftover
nodes (types only connected via the excluded/held-out relation types) via a
post-hoc pass. Produces `outputs/hierarchy.json`. Omit `--embeddings-path`
and `--embeddings-order` to fall back to offline TF-IDF (weaker semantic
signal; see "Semantic embeddings" above).
