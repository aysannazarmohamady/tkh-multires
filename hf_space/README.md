---
title: TKH Hyperedge Embeddings
sdk: gradio
app_file: app.py
---

# TKH hyperedge embeddings (Hugging Face Space)

Computes all-MiniLM-L6-v2 embeddings for every clustering-eligible hyperedge
of the TKH export and returns `semantic_embeddings.npy` +
`embedding_edge_order.json` for `src/method.py` (`--embeddings-path`,
`--embeddings-order`). Live: https://huggingface.co/spaces/aysan98/tkh

The edge filter (`claims`, `authored_by`, `presents` excluded; `cites` held
out; `evaluated_on` merged per article; snapshot membership via
`eff_first_seen`) must match `src/method.py` exactly — enforced by
`tests/test_filter_consistency.py` in the main repo. One file computed on the
full export (cutoff empty or 2026, 334 rows) serves every snapshot.
