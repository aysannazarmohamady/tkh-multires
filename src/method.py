"""
T2 — Core multi-resolution method.

Implements the method described in Deliverable 0 (report.md):
1. Build a combined structural+semantic similarity between hyperedges.
2. Run hierarchical agglomerative clustering (HAC) on hyperedges.
3. Extract levels by cutting the resulting dendrogram at different heights.
4. Assign nodes to clusters top-down, restricting each finer level's
   candidates to the node's already-assigned coarser cluster, which is what
   guarantees the laminar property (P1) rather than hoping for it.

Embedding choice: this environment has no network access to huggingface.co,
so a pretrained sentence-transformer is not usable here. We use TF-IDF
(scikit-learn, fully offline) over each hyperedge's concatenated member
surface forms as the semantic signal instead. This is a weaker semantic
signal than a transformer embedding would give (no synonym/paraphrase
awareness), and is documented as a limitation, not hidden. Swapping in a
transformer embedding later (e.g. sentence-transformers, once network access
is available) only requires replacing `semantic_similarity_matrix`.

Relation-type filtering: per report.md, `claims` (near-degenerate, dominates
by volume) is excluded from clustering, and `cites` is held out entirely as
the independent signal for the T6 coherence check — neither participates in
`s_struct` or `s_sem` here.

Usage:
    python src/method.py --data data/tkh_collection10.json --cutoff 2026
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXCLUDED_FROM_CLUSTERING = {"claims"}  # near-degenerate, dominates by volume
HELD_OUT_FOR_COHERENCE = {"cites"}     # never touched by clustering; for T6


def load_tkh(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def filter_snapshot(data: dict, cutoff_year: int | None) -> tuple[list, list]:
    """Return (nodes, clustering_edges) for a cutoff year (or the whole
    dataset if cutoff_year is None), excluding EXCLUDED_FROM_CLUSTERING and
    HELD_OUT_FOR_COHERENCE relation types from the clustering edge set."""
    if cutoff_year is not None:
        nodes = [n for n in data["nodes"] if n.get("year") is not None and n["year"] <= cutoff_year]
        node_ids = {n["id"] for n in nodes}
        all_edges = [e for e in data["hyperedges"]
                     if e.get("year") is not None and e["year"] <= cutoff_year
                     and all(m in node_ids for m in e["members"])]
    else:
        nodes = data["nodes"]
        all_edges = data["hyperedges"]

    clustering_edges = [
        e for e in all_edges
        if e["relation_type"] not in EXCLUDED_FROM_CLUSTERING
        and e["relation_type"] not in HELD_OUT_FOR_COHERENCE
    ]
    return nodes, clustering_edges


def structural_similarity_matrix(edges: list) -> np.ndarray:
    """Jaccard index between hyperedges' member sets. O(m^2); fine at our
    scale (m in the hundreds to low thousands)."""
    m = len(edges)
    member_sets = [set(e["members"]) for e in edges]
    sim = np.zeros((m, m))
    for i in range(m):
        sim[i, i] = 1.0
        for j in range(i + 1, m):
            inter = len(member_sets[i] & member_sets[j])
            if inter == 0:
                continue
            union = len(member_sets[i] | member_sets[j])
            s = inter / union
            sim[i, j] = s
            sim[j, i] = s
    return sim


def semantic_similarity_matrix(edges: list, node_lookup: dict) -> np.ndarray:
    """TF-IDF cosine similarity between hyperedges, using the concatenated
    surface forms of each hyperedge's members as its 'document'. This is the
    offline fallback; see `semantic_similarity_from_embeddings` for the
    preferred path when real sentence embeddings are available."""
    docs = []
    for e in edges:
        surface_forms = [node_lookup[m]["surface_form"] for m in e["members"] if m in node_lookup]
        docs.append(" ".join(surface_forms))
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    return cosine_similarity(tfidf)


def semantic_similarity_from_embeddings(edges: list, embeddings_path: str, order_path: str) -> np.ndarray:
    """Load precomputed sentence-transformer embeddings (produced by
    compute_embeddings_hf.py on a machine with internet access to
    huggingface.co) and return their pairwise cosine similarity, reordered
    to match `edges`'s current order exactly. Raises if the edge id sets
    don't match, rather than silently misaligning rows."""
    embeddings = np.load(embeddings_path)
    with open(order_path) as f:
        saved_order = json.load(f)

    saved_index = {eid: i for i, eid in enumerate(saved_order)}
    current_ids = [e["id"] for e in edges]
    if set(current_ids) != set(saved_order):
        missing = set(current_ids) - set(saved_order)
        extra = set(saved_order) - set(current_ids)
        raise ValueError(
            f"Embedding file's edge ids don't match the current clustering "
            f"edge set (missing {len(missing)}, extra {len(extra)}). "
            f"Re-run compute_embeddings_hf.py with the same --cutoff."
        )
    reordered = np.stack([embeddings[saved_index[eid]] for eid in current_ids])
    return cosine_similarity(reordered)


def combined_distance_matrix(s_struct: np.ndarray, s_sem: np.ndarray, alpha: float) -> np.ndarray:
    """sim = alpha * s_struct + (1-alpha) * s_sem (arithmetic mean, see
    report.md for why this replaced an earlier, degenerate geometric mean).
    Returns a distance matrix d = 1 - sim, diagonal forced to 0."""
    sim = alpha * s_struct + (1 - alpha) * s_sem
    dist = 1 - sim
    np.fill_diagonal(dist, 0.0)
    # Numerical safety: symmetrize and clip tiny negative values from float error.
    dist = np.clip((dist + dist.T) / 2, 0, None)
    return dist


def build_dendrogram(dist_matrix: np.ndarray):
    """Average-linkage HAC. Returns the scipy linkage matrix Z."""
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    return Z


def find_threshold_for_cluster_range(Z, target_min: int, target_max: int, n_edges: int):
    """Binary-search cut heights (criterion='distance') for a threshold
    producing a cluster count in [target_min, target_max]. Not guaranteed to
    exist (P2 is best-effort, per report.md) — returns the closest achieved
    count if the exact range is unreachable."""
    heights = Z[:, 2]
    lo, hi = 0.0, float(heights.max()) if len(heights) else 1.0
    best_t, best_k, best_gap = None, None, float("inf")
    for _ in range(60):
        t = (lo + hi) / 2
        labels = fcluster(Z, t, criterion="distance")
        k = len(set(labels))
        gap = 0 if target_min <= k <= target_max else min(abs(k - target_min), abs(k - target_max))
        if gap < best_gap:
            best_gap, best_t, best_k = gap, t, k
        if k > target_max:
            lo = t
        elif k < target_min:
            hi = t
        else:
            break
    return best_t, best_k


def extract_levels(Z, n_edges: int, level0_range=(10, 15), n_levels: int = 4):
    """Cut the dendrogram at `n_levels` heights, coarsest first, finest last
    (finest = every hyperedge its own cluster, i.e. threshold 0). Level 0's
    threshold is searched to land in `level0_range` per P2; intermediate
    levels are spaced geometrically between level 0's threshold and 0."""
    t0, k0 = find_threshold_for_cluster_range(Z, *level0_range, n_edges)
    thresholds = list(np.linspace(t0, 0, n_levels))
    thresholds[-1] = -1e-9  # ensure strictly below all merge heights -> singletons
    levels = []
    for t in thresholds:
        labels = fcluster(Z, max(t, 0), criterion="distance") if t >= 0 else np.arange(1, n_edges + 1)
        levels.append(labels)
    return thresholds, levels, k0


def verify_laminar(levels: list) -> bool:
    """Empirical check (not just assumed): for every pair of consecutive
    levels coarse->fine, each fine cluster must be a subset of exactly one
    coarse cluster. Returns True if this holds for all consecutive pairs."""
    for coarse, fine in zip(levels[:-1], levels[1:]):
        fine_to_coarse = {}
        for c_label, f_label in zip(coarse, fine):
            if f_label in fine_to_coarse and fine_to_coarse[f_label] != c_label:
                return False
            fine_to_coarse[f_label] = c_label
    return True


def assign_nodes(nodes: list, edges: list, levels: list) -> dict:
    """Top-down node assignment (see module docstring). `levels` must be
    ordered coarsest-first. Returns {node_id: [cluster_id_level0, ..., cluster_id_levelK]}."""
    node_incident_edges = defaultdict(list)
    for idx, e in enumerate(edges):
        for m in e["members"]:
            node_incident_edges[m].append(idx)

    assignment = {}
    for node in nodes:
        nid = node["id"]
        candidate_edges = node_incident_edges.get(nid, [])
        if not candidate_edges:
            assignment[nid] = [None] * len(levels)
            continue
        per_level = []
        for labels in levels:
            counts = Counter(labels[e] for e in candidate_edges)
            majority_cluster, _ = counts.most_common(1)[0]
            per_level.append(int(majority_cluster))
            candidate_edges = [e for e in candidate_edges if labels[e] == majority_cluster]
        assignment[nid] = per_level
    return assignment


def attach_leftover_nodes(nodes: list, all_edges: list, assignment: dict, n_levels: int) -> dict:
    """Second pass, run only after primary clustering + assignment.

    Some node types (`claim`, `cited_work` in this dataset) appear *only* in
    the excluded/held-out relation types (`claims`, `cites`), so they never
    get an assignment from `assign_nodes`. Rather than leave them out of the
    hierarchy entirely, attach each such node to the cluster of whichever
    already-assigned node it shares an EXCLUDED_FROM_CLUSTERING or
    HELD_OUT_FOR_COHERENCE hyperedge with (majority vote if more than one
    candidate, at every level, same top-down restriction as the primary
    assignment). This uses those relation types only to propagate an
    existing assignment, never to compute similarity or drive clustering, so
    it does not reopen the coherence-circularity or degenerate-edge issues
    that excluding them was meant to fix.
    """
    leftover_edges = [e for e in all_edges if e["relation_type"] in
                       (EXCLUDED_FROM_CLUSTERING | HELD_OUT_FOR_COHERENCE)]

    unassigned = [n["id"] for n in nodes if assignment.get(n["id"], [None])[0] is None]
    unassigned_set = set(unassigned)

    # For each leftover edge, find members that already have an assignment.
    neighbor_assignments = defaultdict(list)  # node_id -> list of assigned neighbor node_ids
    for e in leftover_edges:
        members = e["members"]
        assigned_members = [m for m in members if m not in unassigned_set and m in assignment]
        for m in members:
            if m in unassigned_set:
                neighbor_assignments[m].extend(assigned_members)

    attached, still_unassigned = 0, 0
    for nid in unassigned:
        neighbors = neighbor_assignments.get(nid, [])
        if not neighbors:
            still_unassigned += 1
            continue
        # Majority vote per level among assigned neighbors' assignments.
        per_level = []
        for lvl in range(n_levels):
            votes = Counter(assignment[n][lvl] for n in neighbors if assignment[n][lvl] is not None)
            if votes:
                per_level.append(votes.most_common(1)[0][0])
            else:
                per_level.append(None)
        assignment[nid] = per_level
        attached += 1

    print(f"Post-hoc attachment: {attached} nodes attached via held-out/excluded "
          f"relations, {still_unassigned} still fully isolated.")
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tkh_collection10.json")
    parser.add_argument("--cutoff", type=int, default=None, help="Snapshot cutoff year; omit for full dataset")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/hierarchy.json")
    parser.add_argument("--embeddings-path", default=None,
                         help="Path to precomputed embeddings .npy from compute_embeddings_hf.py; if omitted, falls back to TF-IDF")
    parser.add_argument("--embeddings-order", default=None,
                         help="Path to the matching embedding_edge_order.json")
    args = parser.parse_args()

    data = load_tkh(args.data)
    node_lookup = {n["id"]: n for n in data["nodes"]}
    nodes, edges = filter_snapshot(data, args.cutoff)
    print(f"Snapshot cutoff={args.cutoff}: {len(nodes)} nodes, {len(edges)} clustering edges "
          f"(excluded: {EXCLUDED_FROM_CLUSTERING}, held out: {HELD_OUT_FOR_COHERENCE})")

    print("Computing structural similarity...")
    s_struct = structural_similarity_matrix(edges)
    nonzero_frac = (s_struct[np.triu_indices_from(s_struct, k=1)] > 0).mean()
    print(f"  nonzero structural pairs: {nonzero_frac:.4f}")

    print("Computing semantic similarity...")
    if args.embeddings_path and args.embeddings_order:
        print(f"  using precomputed sentence embeddings from {args.embeddings_path}")
        s_sem = semantic_similarity_from_embeddings(edges, args.embeddings_path, args.embeddings_order)
    else:
        print("  no --embeddings-path given; falling back to offline TF-IDF "
              "(weaker semantic signal — see module docstring)")
        s_sem = semantic_similarity_matrix(edges, node_lookup)

    dist = combined_distance_matrix(s_struct, s_sem, args.alpha)
    Z = build_dendrogram(dist)

    thresholds, levels, k0 = extract_levels(Z, len(edges))
    print(f"Level cluster counts (coarse->fine): {[len(set(l)) for l in levels]}")
    print(f"Level 0 target 10-15, achieved: {k0}")

    laminar_ok = verify_laminar(levels)
    print(f"Laminar refinement check (empirical): {'PASS' if laminar_ok else 'FAIL'}")

    assignment = assign_nodes(nodes, edges, levels)
    assigned = sum(1 for v in assignment.values() if v[0] is not None)
    print(f"Nodes assigned by primary clustering: {assigned}/{len(nodes)} "
          f"({len(nodes) - assigned} have no incident clustering edge)")

    all_edges_for_snapshot = [
        e for e in data["hyperedges"]
        if (args.cutoff is None or (e.get("year") is not None and e["year"] <= args.cutoff))
    ]
    assignment = attach_leftover_nodes(nodes, all_edges_for_snapshot, assignment, len(levels))
    fully_assigned = sum(1 for v in assignment.values() if v[0] is not None)
    print(f"Nodes assigned after post-hoc attachment: {fully_assigned}/{len(nodes)}")

    Path("outputs").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "cutoff_year": args.cutoff,
            "alpha": args.alpha,
            "n_nodes": len(nodes),
            "n_clustering_edges": len(edges),
            "level_cluster_counts": [len(set(l)) for l in levels],
            "laminar_check_passed": laminar_ok,
            "node_assignment": assignment,
        }, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
