"""
Hugging Face Space app: upload the TKH json export, get back the two files
needed by the main pipeline's src/method.py (--embeddings-path /
--embeddings-order) to use real sentence-transformer embeddings instead of
the offline TF-IDF fallback.

IMPORTANT: this filtering/merging logic must stay in sync with
src/method.py's filter_snapshot / merge_evaluated_on_by_article, or the
embedded edge ids won't line up with what method.py looks for.
"""

import json
import tempfile

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(fn):
        return fn

EXCLUDED_FROM_CLUSTERING = {"claims", "authored_by", "presents"}  # keep in sync with src/method.py
HELD_OUT_FOR_COHERENCE = {"cites"}
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def merge_evaluated_on_by_article(edges):
    """Must match src/method.py's merge_evaluated_on_by_article exactly."""
    by_article = {}
    others = []
    for e in edges:
        if e["relation_type"] == "evaluated_on":
            aid = e.get("provenance", {}).get("article_id")
            by_article.setdefault(aid, []).append(e)
        else:
            others.append(e)
    merged = []
    for aid, group in by_article.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        seen, member_union = set(), []
        for e in group:
            for m in e["members"]:
                if m not in seen:
                    seen.add(m)
                    member_union.append(m)
        merged.append({
            "id": f"h_merged_eval_{aid}",
            "relation_type": "evaluated_on",
            "members": member_union,
            "provenance": group[0].get("provenance"),
        })
    return others + merged


def compute_eff_first_seen(data):
    """Must match src/load_graph.py's compute_eff_first_seen exactly:
    min(node's first_seen_year, earliest incident edge's article_year)."""
    earliest = {}
    for e in data["hyperedges"]:
        ay = e.get("provenance", {}).get("article_year")
        if ay is None:
            continue
        for m in e["members"]:
            if m not in earliest or ay < earliest[m]:
                earliest[m] = ay
    eff = {}
    for n in data["nodes"]:
        fsy, eey = n.get("first_seen_year"), earliest.get(n["id"])
        eff[n["id"]] = eey if fsy is None else (fsy if eey is None else min(fsy, eey))
    return eff


def filter_clustering_edges(data, cutoff_year=None):
    """Must match src/method.py's filter_snapshot (edge part) exactly;
    enforced by tests/test_filter_consistency.py."""
    if cutoff_year is not None:
        eff = compute_eff_first_seen(data)
        node_ids = {n["id"] for n in data["nodes"]
                    if eff.get(n["id"]) is not None and eff[n["id"]] <= cutoff_year}
        all_edges = [e for e in data["hyperedges"]
                     if e.get("provenance", {}).get("article_year") is not None
                     and e["provenance"]["article_year"] <= cutoff_year
                     and all(m in node_ids for m in e["members"])]
    else:
        all_edges = data["hyperedges"]

    clustering_edges = [e for e in all_edges
                         if e["relation_type"] not in EXCLUDED_FROM_CLUSTERING
                         and e["relation_type"] not in HELD_OUT_FOR_COHERENCE]
    return merge_evaluated_on_by_article(clustering_edges)


@GPU_DECORATOR
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
        f"(excluded: {sorted(EXCLUDED_FROM_CLUSTERING)}, held out: {HELD_OUT_FOR_COHERENCE}, "
        f"evaluated_on merged per-article) with {MODEL_NAME}. Shape: {embeddings.shape}."
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
        "for every clustering-eligible hyperedge (excludes `claims`, "
        "`authored_by`, `presents`; merges `evaluated_on` rows per article; "
        "matching src/method.py in the main repo). Download both output "
        "files and place them in the main repo's outputs/ folder."
    ),
)

if __name__ == "__main__":
    demo.launch()
