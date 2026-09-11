#!/usr/bin/env bash
# Full pipeline, in dependency order. Run from the repo root:  bash reproduce.sh
# Total runtime ~5 min (the T3 null, 4 x 1000 re-clusterings, dominates).
set -euo pipefail
EMB="--embeddings-path outputs/semantic_embeddings.npy --embeddings-order outputs/embedding_edge_order.json"

python tests/test_filter_consistency.py                    # embedding scripts == method.py filter
python src/load_graph.py --data data/tkh_collection10.json # T1
for y in 2020 2022 2024; do
  python src/method.py --cutoff $y $EMB --out outputs/hierarchy_$y.json   # T2
done
python src/method.py --cutoff 2026 $EMB --out outputs/hierarchy.json
python src/hyperedge_collapse.py --all                     # T4 (all levels, all snapshots)
python src/build_temporal_events.py                        # T3 identity + event log
python src/t3_perturbation_real_embeddings.py              # T3 stability, snapshot-matched nulls
python src/sensitivity_diagnostics.py                      # documented method sensitivities
python src/paper_identity_diagnostic.py                    # T2 diagnostic
python src/extrinsic_eval.py                               # T6 extrinsic
for y in 2020 2022 2024 2026; do                           # T5 labeller inputs (labels themselves are an LLM step)
  h=outputs/hierarchy_$y.json; [ $y = 2026 ] && h=outputs/hierarchy.json
  for l in 0 1; do
    python src/prepare_labelling_input.py --hierarchy $h --cutoff $y --level $l \
        --out outputs/labelling_input_${y}_level$l.json > /dev/null
  done
done
python src/assemble_metrics.py                             # T6 metrics.json
python src/merge_supernodes.py --template-unlabelled --strict   # §6.2 schema, LAST hierarchy write
python src/make_figures.py
echo "Pipeline complete."
