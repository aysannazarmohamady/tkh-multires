"""
Hugging Face Space app: upload the TKH json export, get back the two files
needed by the main pipeline's src/method.py (--embeddings-path /
--embeddings-order) to use real sentence-transformer embeddings instead of
the offline TF-IDF fallback.
"""

import json
import tempfile

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

EXCLUDED_FROM_CLUSTERING = {"claims"}
HELD_OUT_FOR_COHERENCE = {"cites"}
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def filter_clustering_edges(data, cutoff_year=None):
    if cutoff_year:
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


def compute_embeddings(file_obj, cutoff_year):
    with open(file_obj.name) as f:
        data = json.load(f)

    node_lookup = {n["id"]: n for n in data["nodes"]}
    cutoff = int(cutoff_year) if cutoff_year else None
    edges = filter_clustering_edges(data, cutoff)

    docs = []
    for e in edges:
        surface_forms = [node_lookup[m]["surface_form"] for m in e["members"] if m in node_lookup]
        docs.append(", ".join(surface_forms))

    model = get_model()
    embeddings = model.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    emb_path = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    order_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    np.save(emb_path, embeddings)
    with open(order_path, "w") as f:
        json.dump([e["id"] for e in edges], f)

    status = (
        f"Embedded {len(edges)} clustering-eligible hyperedges "
        f"(excluded: {EXCLUDED_FROM_CLUSTERING}, held out: {HELD_OUT_FOR_COHERENCE}) "
        f"with {MODEL_NAME}. Shape: {embeddings.shape}."
    )
    return status, emb_path, order_path


demo = gr.Interface(
    fn=compute_embeddings,
    inputs=[
        gr.File(label="tkh_collection10.json"),
        gr.Textbox(label="Cutoff year (optional, e.g. 2026)", value=""),
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.File(label="semantic_embeddings.npy"),
        gr.File(label="embedding_edge_order.json"),
    ],
    title="TKH Hyperedge Embeddings",
    description=(
        "Upload the TKH export to compute sentence-transformer embeddings "
        "for every clustering-eligible hyperedge (excludes `claims` and "
        "`cites`, matching src/method.py in the main repo). Download both "
        "output files and place them in the main repo's outputs/ folder."
    ),
)

if __name__ == "__main__":
    demo.launch()
