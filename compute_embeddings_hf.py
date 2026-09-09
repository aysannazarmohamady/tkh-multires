"""
Standalone embedding script — run this on Hugging Face (or any environment
with internet access to huggingface.co), NOT in the main pipeline's sandbox,
which has no network access there.

Computes a sentence-transformer embedding for every clustering-eligible
hyperedge (same filtering as src/method.py: excludes `claims`, `cites`), and
saves the embeddings + the exact edge id order to two files. Bring both
files back into the main repo's `outputs/` folder; `src/method.py` will pick
them up automatically if present (see its --embeddings-path option) and use
them instead of the TF-IDF fallback.

Usage (on a machine/notebook with internet + `pip install sentence-transformers`):
    python compute_embeddings_hf.py --data tkh_collection10.json

Outputs:
    semantic_embeddings.npy       — float32 array, shape (n_clustering_edges, dim)
    embedding_edge_order.json     — list of hyperedge ids, in the same row order
"""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

EXCLUDED_FROM_CLUSTERING = {"claims"}
HELD_OUT_FOR_COHERENCE = {"cites"}

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good general-purpose default


def load_tkh(path):
    with open(path) as f:
        return json.load(f)


def filter_clustering_edges(data, cutoff_year=None):
    if cutoff_year is not None:
        nodes = [n for n in data["nodes"] if n.get("year") is not None and n["year"] <= cutoff_year]
        node_ids = {n["id"] for n in nodes}
        all_edges = [e for e in data["hyperedges"]
                     if e.get("year") is not None and e["year"] <= cutoff_year
                     and all(m in node_ids for m in e["members"])]
    else:
        all_edges = data["hyperedges"]

    return [e for e in all_edges
            if e["relation_type"] not in EXCLUDED_FROM_CLUSTERING
            and e["relation_type"] not in HELD_OUT_FOR_COHERENCE]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tkh_collection10.json")
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--out-embeddings", default="semantic_embeddings.npy")
    parser.add_argument("--out-order", default="embedding_edge_order.json")
    args = parser.parse_args()

    data = load_tkh(args.data)
    node_lookup = {n["id"]: n for n in data["nodes"]}
    edges = filter_clustering_edges(data, args.cutoff)
    print(f"{len(edges)} clustering-eligible hyperedges to embed.")

    docs = []
    for e in edges:
        surface_forms = [node_lookup[m]["surface_form"] for m in e["members"] if m in node_lookup]
        docs.append(", ".join(surface_forms))

    print(f"Loading model {MODEL_NAME} (requires internet on first run)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding...")
    embeddings = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(args.out_embeddings, embeddings)
    with open(args.out_order, "w") as f:
        json.dump([e["id"] for e in edges], f)

    print(f"Saved {embeddings.shape} embeddings to {args.out_embeddings}")
    print(f"Saved edge id order to {args.out_order}")
    print("Download both files and place them in the main repo's outputs/ folder.")


if __name__ == "__main__":
    main()
