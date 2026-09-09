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
  - [ ] Method implementation (hyperedge clustering + node assignment)
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

## Reproduction

```bash
python src/load_graph.py --data data/tkh_collection10.json
```

Produces per-snapshot descriptive stats (node/edge counts, node-type
distribution, hyper-edge arity distribution, degree distribution and
singleton fraction, growth between snapshots by type, duplicate
surface-form detection, and data-quality notes) in
`outputs/t1_snapshot_stats.json`.
