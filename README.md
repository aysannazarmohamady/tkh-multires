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
        see `src/method.py`; verified on all 4 snapshots. Signals are
        rank-normalized before combining (fixes an alpha=0.5 scale
        mismatch that made the method ~95% semantic in practice; rank
        normalization raises the structural variance share to only 8.7%,
        an honest, only partially fixable limit given 96.7% of structural
        pairs are exactly zero — see `report.md`). Coverage and level-0
        balance vary by snapshot (this is reported honestly, not smoothed
        over — see `report.md`, "Second/Third review round"):

        | cutoff | nodes | fully unassigned | level-0 top-2 concentration |
        |---|---|---|---|
        | 2020 | 1,839 | 364 (19.8%) | 33.2% |
        | 2022 | 2,496 | 357 (14.3%) | 36.5% |
        | 2024 | 4,307 | 211 (4.9%) | 58.4% |
        | 2026 | 5,798 | 0 (0.0%) | 47.7% |

        The 2020 unassigned nodes have no incident edge among the clustered
        or held-out/excluded relation types *at that cutoff* (their only
        edges appear later) — this is a real property of the growing
        corpus, not a bug; P1/P2 hold over the assigned subgraph at each
        snapshot, not over isolated nodes that aren't in any hyperedge yet.
  - [x] Coherence probe (bibliographic coupling between articles, with a
        provenance assertion closing the earlier leak) — see
        `src/coherence_probe.py` and `report.md`, "Third review round" and
        "T2 correction". **Result updated** after the rank-normalization
        fix changed the clustering: significant coherence signal at 2024
        (z=2.84, p=0.0068) and 2026 (z=2.59, p=0.011); not significant at
        2020/2022 (too few articles). An earlier draft reported this as a
        null result under the pre-fix clustering — flagged and corrected,
        not silently updated.
- [x] T3 — Temporal coupling — see `src/temporal_reg.py` (mechanism sweep +
      decisive perturbation test + null model) and
      `src/build_temporal_events.py` (event log). **We found and fixed a
      circularity bug in our own stability metric**: an initial
      regularization mechanism (lambda=0.2) directly manipulated the same
      ARI-vs-previous-snapshot metric used to evaluate it, and separately
      bypassed the rank-normalization fix below. After fixing both and
      re-selecting lambda by perturbation-ARI (not transition-ARI, which
      the mechanism manipulates), the evidence-based choice is
      **lambda=0 — no regularization**: it has the highest
      perturbation-ARI (0.457) of any value tested, confirmed by a null
      model (shuffled prior labels) that is numerically identical to the
      observed result at lambda=0, as it should be. Real cross-snapshot
      ARI (0.43-0.53, mean 0.49) is reported honestly; identity is tracked
      via post-hoc matching, not artificially enforced. Full account in
      `report.md`, "T3 — Temporal coupling: circularity found and fixed."
- [x] T4 — Hyper-edge collapse — see `src/hyperedge_collapse.py`. Genuine
      hypergraph-native coarsening (arity>=3 relations kept as hyperedges
      between super-nodes, not clique-expanded); a `clique_expand=True`
      mode is also implemented for direct projection-loss comparison
      (62 native hyperedges -> 83 edges if clique-expanded, see `report.md`).
      Hand-verified against a real arity-20 hyperedge.
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
hierarchical agglomerative clustering (**complete linkage by default** —
average-linkage was found to chain into two dominant clusters holding 75%
of all nodes; see `report.md`), extracts a level-0 cut targeting 10-15
clusters (P2) plus geometrically-spaced intermediate levels, assigns nodes
top-down so the result is laminar by construction (P1, empirically verified
each run), and attaches leftover nodes via a post-hoc pass, tagging each
with a `provenance` (`primary` / `claims` / `cites`). Produces
`outputs/hierarchy.json`. Omit `--embeddings-path` / `--embeddings-order`
to fall back to offline TF-IDF. The same embeddings file (computed once on
the full 2026 snapshot) can be reused for any earlier `--cutoff`, since a
hyperedge's embedding only depends on its own members, not the snapshot.

```bash
python src/coherence_probe.py outputs/hierarchy_2020.json \
    outputs/hierarchy_2024.json outputs/hierarchy.json
```

Runs the T6 coherence probe (bibliographic coupling between articles via
held-out `cites` edges, with a `provenance == "primary"` assertion that
fails loudly if the independence guarantee is ever broken). Prints one JSON
result line per snapshot file.

```bash
python src/temporal_reg.py
```

Runs the lambda sweep (in units of the distance distribution's std),
selects `lambda_sd` by perturbation-ARI (not transition-ARI, which an
earlier version showed the regularizer directly manipulates), and runs the
shuffled-prior null model. Saves
`outputs/lambda_selection_and_null.json`. Current evidence-based choice:
`lambda_sd=0` (no regularization).

```bash
python src/build_temporal_events.py
```

Runs the chain across all 4 snapshots at the selected `lambda_sd=0` and
produces `outputs/temporal_events.json` (persistent cluster identity +
birth/growth/merge/split/dissolution event log, level 0, via post-hoc
hyperedge-Jaccard matching only).

```bash
python src/hyperedge_collapse.py --hierarchy outputs/hierarchy.json --level 0
```

Runs T4: collapses all hyperedges (including `claims`/`cites`) according to
how their endpoints are distributed across level-0 super-nodes, producing
`outputs/hyperedge_collapse.json` (internal edges + cohesion score per
cluster, native hypergraph-preserving coarse edges, and a clique-expanded
version for direct projection-loss comparison).
