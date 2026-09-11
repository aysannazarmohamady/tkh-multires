"""
T2 — Core multi-resolution method.

Implements the method described in Deliverable 0 (report.md):
1. Build a combined structural+semantic similarity between hyperedges.
2. Run hierarchical agglomerative clustering (HAC) on hyperedges.
3. Extract levels by cutting the resulting dendrogram at different heights.
4. Assign nodes to clusters top-down, restricting each finer level's
   candidates to the node's already-assigned coarser cluster, which is what
   guarantees the laminar property (P1) rather than hoping for it.

Semantic signal: the shipped hierarchies use real sentence embeddings
(all-MiniLM-L6-v2, 384-d) precomputed once on the 2026 clustering-edge set
by `compute_embeddings_hf.py` / `hf_space/app.py` (this sandbox has no
network access to huggingface.co) and passed in via `--embeddings-path` /
`--embeddings-order`. Without those flags the script falls back to offline
TF-IDF over each hyperedge's concatenated member surface forms — a weaker
signal (no synonym/paraphrase awareness, correlates ~0.69 with the
structural signal vs ~0.31 for MiniLM), kept only as a fallback.

Relation-type filtering: `claims`, `authored_by` and `presents` are
excluded from clustering (EXCLUDED_FROM_CLUSTERING) and `cites` is held out
entirely as the independent signal for the T6 coherence check
(HELD_OUT_FOR_COHERENCE). None of them enter `s_struct` or `s_sem`; their
nodes are placed afterwards by `place_articles_via_presents` and
`attach_leftover_nodes`, with a per-node `provenance` tag.

Known sensitivities (documented, deliberately NOT changed — changing them
re-clusters every snapshot and invalidates all labels; see report.md,
"Limitations"): (a) rank normalisation gives all zero-overlap structural
pairs one tied mid-rank, so structure acts mostly as a binary "any overlap"
bonus; (b) majority-vote ties in node assignment are broken by edge-list
order. Both are quantified by `src/sensitivity_diagnostics.py`.

Usage:
    python src/method.py --data data/tkh_collection10.json --cutoff 2026 \
        --embeddings-path outputs/semantic_embeddings.npy \
        --embeddings-order outputs/embedding_edge_order.json
"""

import argparse
import math
import sys
sys.path.insert(0, "src")
from load_graph import compute_eff_first_seen
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXCLUDED_FROM_CLUSTERING = {"claims", "authored_by", "presents"}  # near-degenerate volume (claims), pure identity (authored_by), or contributes nothing to similarity and only anchors article placement (presents; arity 2, see place_articles_via_presents)
HELD_OUT_FOR_COHERENCE = {"cites"}     # never touched by clustering; for T6
# Post-hoc attachment priority; `cites` last so it is used only as a last resort.
ATTACH_RELATION_ORDER = ["claims", "authored_by", "cites"]
assert set(ATTACH_RELATION_ORDER) >= (EXCLUDED_FROM_CLUSTERING | HELD_OUT_FOR_COHERENCE) - {"presents"}


def load_tkh(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def merge_evaluated_on_by_article(edges: list) -> list:
    """Merge each article's separate `evaluated_on` edges (one per
    result-table row in the source paper) into a single hyperedge per
    article.

    Fix (found by external review, verified against our data): 301
    `evaluated_on` edges came from only 36 distinct articles (one article
    had 33 separate `evaluated_on` edges). Since hyperedge similarity is
    driven by member overlap, this let many near-duplicate rows from the
    SAME paper dominate the similarity signal, pulling level-0 clusters
    toward "papers" rather than "ideas" (measured: NMI between level-0
    cluster and source article was 0.498-0.61 depending on measurement
    method — clusters were substantially explainable by "which paper", not
    just topic). Merging each article's rows into one hyperedge (union of
    members) removes this specific source of paper-identity leakage while
    keeping every original member.
    """
    by_article = defaultdict(list)
    others = []
    for e in edges:
        if e["relation_type"] == "evaluated_on":
            aid = e.get("provenance", {}).get("article_id")
            by_article[aid].append(e)
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
            "attributes": {"merged_from_edge_ids": [e["id"] for e in group]},
        })
    return others + merged


def filter_snapshot(data: dict, cutoff_year: int | None) -> tuple[list, list]:
    """Return (nodes, clustering_edges) for a cutoff year (or the whole
    dataset if cutoff_year is None), excluding EXCLUDED_FROM_CLUSTERING and
    HELD_OUT_FOR_COHERENCE relation types from the clustering edge set.

    Temporal-leak fix (found by external review, same issue as in
    load_graph.py): a node enters a snapshot when
    `eff_first_seen = min(first_seen_year, earliest incident edge's
    article_year) <= cutoff` (when the corpus first recorded it), never on
    `year`/`origin_year` (when the entity was invented in the real world).
    Edges are filtered on `provenance.article_year`. An earlier version used
    `year` for both, leaking 334/332/143 "future" nodes into the
    2020/2022/2024 snapshots.

    `authored_by` is excluded from clustering: it is a pure identity edge
    with no topical content, and including it pulled clusters toward "same
    paper" rather than "same idea". Its author nodes are placed afterwards
    by `attach_leftover_nodes` (provenance "authored_by").

    `presents` is also excluded: it is arity-2 {article, method}, so its
    weighted-Jaccard overlap with other edges is almost always zero and it
    contributes nothing to similarity. Articles are instead placed
    deterministically by `place_articles_via_presents` (they inherit the
    path of the method they present), which keeps article placement
    independent of `cites` and therefore of the T6 coherence probe.
    (An intermediate version excluded `presents` WITHOUT that placement
    step, which routed articles through `cites` and tripped the coherence
    probe's provenance assertion; that is why the placement step exists.)
    """
    if cutoff_year is not None:
        eff_first_seen = compute_eff_first_seen(data)
        nodes = [n for n in data["nodes"]
                 if eff_first_seen.get(n["id"]) is not None and eff_first_seen[n["id"]] <= cutoff_year]
        node_ids = {n["id"] for n in nodes}
        all_edges = [e for e in data["hyperedges"]
                     if e.get("provenance", {}).get("article_year") is not None
                     and e["provenance"]["article_year"] <= cutoff_year
                     and all(m in node_ids for m in e["members"])]
    else:
        nodes = data["nodes"]
        all_edges = data["hyperedges"]

    clustering_edges = [
        e for e in all_edges
        if e["relation_type"] not in EXCLUDED_FROM_CLUSTERING
        and e["relation_type"] not in HELD_OUT_FOR_COHERENCE
    ]
    clustering_edges = merge_evaluated_on_by_article(clustering_edges)
    return nodes, clustering_edges


def structural_similarity_matrix(edges: list) -> np.ndarray:
    """Paper-frequency-weighted Jaccard between hyperedges' member sets.
    O(m^2); fine at our scale.

    Correction (second external review): an earlier version weighted by
    edge-frequency IDF (`log(n_edges/df(v))`), which barely reduced
    paper-identity bias, because the real driver isn't hub NODES in
    general — it's a single hub node PER PAPER (that paper's central
    method node), which co-occurs with nearly all of that paper's edges
    regardless of what they're actually about. Verified directly: one
    sample article had 39 clustering-eligible edges, 38 of which shared
    the same method node. Edge-frequency IDF barely touches this because
    the hub node's edge-count IS mostly "how many edges this one paper
    contributed", not a generic popularity signal.

    Fix, corrected once during implementation: weight each member `v` by
    `w(v) = log(1 + n_papers(v))`, where `n_papers(v)` is the number of
    DISTINCT ARTICLES whose edges mention `v`. An initial version used the
    reciprocal, `1/log(1+n_papers(v))` — this was a sign error: it gave
    single-paper-specific nodes (a paper's own idiosyncratic hub) MORE
    weight, not less, and measurably made paper-identity NMI worse (0.612
    vs 0.566 unweighted) before being caught and inverted. The corrected
    direction downweights nodes specific to one paper and upweights nodes
    that genuinely recur across many different papers (real shared
    concepts), which is the intended effect.
    """
    m = len(edges)
    member_sets = [set(e["members"]) for e in edges]

    node_papers = defaultdict(set)
    for e in edges:
        aid = e.get("provenance", {}).get("article_id")
        if aid is None:
            continue
        for v in e["members"]:
            node_papers[v].add(aid)
    weight = {v: np.log(1 + len(papers)) for v, papers in node_papers.items()}

    sim = np.zeros((m, m))
    for i in range(m):
        sim[i, i] = 1.0
        for j in range(i + 1, m):
            inter = member_sets[i] & member_sets[j]
            if not inter:
                continue
            union = member_sets[i] | member_sets[j]
            # math.fsum is exactly rounded, hence independent of set-iteration
            # order (which depends on PYTHONHASHSEED). Plain sum() gave ~1e-17
            # run-to-run differences that flipped near-tied ranks in ~1% of
            # perturbed re-clusterings (found by the clean-room check).
            w_inter = math.fsum(weight.get(v, 1.0) for v in inter)
            w_union = math.fsum(weight.get(v, 1.0) for v in union)
            s = w_inter / w_union if w_union > 0 else 0.0
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
    compute_embeddings_hf.py / the HF Space on a machine with internet
    access) and return their pairwise cosine similarity, reordered to match
    `edges`'s current order exactly.

    Correction (found by external review): an earlier version required the
    embedding file's edge id set to *exactly* match the current snapshot's
    edge set, which meant every snapshot needed its own Space run (a real
    reproducibility risk, and one earlier snapshots silently failed
    without). A hyperedge's embedding only depends on its own members'
    surface forms, which don't change across snapshots, so an embeddings
    file computed once on the full (2026) edge set is valid for any earlier
    cutoff too — we only need `edges`'s ids to be a SUBSET of the saved
    order, not an exact match, and we select the needed rows by id.
    """
    embeddings = np.load(embeddings_path)
    with open(order_path) as f:
        saved_order = json.load(f)

    saved_index = {eid: i for i, eid in enumerate(saved_order)}
    current_ids = [e["id"] for e in edges]
    missing = [eid for eid in current_ids if eid not in saved_index]
    if missing:
        raise ValueError(
            f"{len(missing)} clustering edge id(s) are not present in the "
            f"embeddings file (e.g. {missing[:3]}). The embeddings file must "
            f"be computed on a snapshot that is a superset of the current "
            f"one (e.g. the full/2026 snapshot covers all earlier cutoffs); "
            f"re-run compute_embeddings_hf.py / the HF Space on a broader "
            f"cutoff if this snapshot introduces edges the file doesn't have."
        )
    reordered = np.stack([embeddings[saved_index[eid]] for eid in current_ids])
    return cosine_similarity(reordered)


def combined_distance_matrix(s_struct: np.ndarray, s_sem: np.ndarray, alpha: float) -> np.ndarray:
    """sim = alpha * s_struct_norm + (1-alpha) * s_sem_norm (arithmetic mean
    of RANK-NORMALIZED signals), returns distance d = 1 - sim.

    Correction (found by external review, verified against our data): the
    raw signals live on very different scales — s_struct has mean 0.0044
    (nonzero on only 3.3% of pairs) while s_sem has mean 0.300. At
    alpha=0.5 on the raw values, the structural term supplied only 4.7% of
    the combined similarity's variance (confirmed: var=0.000268 vs
    0.005472) — the method was effectively ~95% semantic despite alpha=0.5
    being presented as a balanced reconciliation. We rank-normalize each
    signal to [0,1] over its own upper-triangle BEFORE mixing, so alpha
    actually controls the structure/semantics balance as claimed, rather
    than being dominated by whichever raw signal happens to have larger
    variance.
    """
    def rank_normalize(matrix):
        iu = np.triu_indices_from(matrix, k=1)
        vals = matrix[iu]
        ranks = rankdata(vals, method="average") / len(vals)
        out = np.zeros_like(matrix)
        out[iu] = ranks
        out = out + out.T
        np.fill_diagonal(out, 1.0)
        return out

    s_struct_norm = rank_normalize(s_struct)
    s_sem_norm = rank_normalize(s_sem)

    sim = alpha * s_struct_norm + (1 - alpha) * s_sem_norm
    dist = 1 - sim
    np.fill_diagonal(dist, 0.0)
    # Numerical safety: symmetrize and clip tiny negative values from float error.
    dist = np.clip((dist + dist.T) / 2, 0, None)
    return dist


def build_dendrogram(dist_matrix: np.ndarray, method: str = "average"):
    """HAC with a configurable linkage method. Returns the scipy linkage
    matrix Z. Default is average-linkage; see report.md for a comparison
    against complete/ward, which were found to produce more balanced level-0
    clusters on this dataset (average-linkage's tendency to chain produced
    two dominant clusters holding 75% of all nodes)."""
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    return Z


def find_threshold_for_target_k(Z, target_k: int, tolerance: float = 0.2):
    """Binary-search a cut height (criterion='distance') for a threshold
    producing a cluster count within `tolerance` (relative) of `target_k`.
    Returns the closest achieved (threshold, k) if the exact target is
    unreachable (dendrograms can jump cluster counts at a single merge)."""
    heights = Z[:, 2]
    lo, hi = 0.0, float(heights.max()) if len(heights) else 1.0
    best_t, best_k, best_gap = None, None, float("inf")
    for _ in range(60):
        t = (lo + hi) / 2
        labels = fcluster(Z, t, criterion="distance")
        k = len(set(labels))
        gap = abs(k - target_k)
        if gap < best_gap:
            best_gap, best_t, best_k = gap, t, k
        if k > target_k:
            lo = t
        elif k < target_k:
            hi = t
        else:
            break
    return best_t, best_k


def find_threshold_for_cluster_range(Z, target_min: int, target_max: int, n_edges: int):
    """As find_threshold_for_target_k, but for a [target_min, target_max]
    range rather than a single value (used only for level 0's P2 budget)."""
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
    (finest = every hyperedge its own cluster).

    Level 0's threshold is searched to land in `level0_range` per P2.
    Intermediate levels' *target cluster counts* (not thresholds) are
    spaced geometrically between level 0's count and n_edges, then each
    target count is independently searched for via
    `find_threshold_for_target_k`.

    Correction (found by external review): an earlier version spaced the
    *thresholds* linearly (`np.linspace` on height, despite a docstring that
    claimed "geometrically"), which produced a degenerate ladder in
    practice (e.g. 14 -> 356 -> 660 -> 702 clusters on the full snapshot) —
    almost all of the "coarsening" happened in one big jump, leaving no
    usable intermediate level for drill-down. Targeting cluster *counts*
    geometrically and searching each one directly fixes both the
    docstring/code mismatch and the degenerate ladder.
    """
    t0, k0 = find_threshold_for_cluster_range(Z, *level0_range, n_edges)
    target_ks = np.geomspace(max(k0, 2), n_edges, n_levels).round().astype(int)
    target_ks[0] = k0  # keep level 0 exactly what the P2 search found
    target_ks[-1] = n_edges  # finest level: every hyperedge its own cluster

    levels = []
    for i, k in enumerate(target_ks):
        if i == len(target_ks) - 1:
            labels = np.arange(1, n_edges + 1)
        else:
            t, achieved_k = find_threshold_for_target_k(Z, int(k))
            labels = fcluster(Z, t, criterion="distance")
        levels.append(labels)
    return target_ks, levels, k0


def verify_laminar(levels: list) -> bool:
    """Empirical check (not just assumed): for every pair of consecutive
    HYPEREDGE-cluster levels coarse->fine, each fine cluster must be a
    subset of exactly one coarse cluster. Returns True if this holds for
    all consecutive pairs.

    NOTE: this checks the hyperedge dendrogram cuts only. It does NOT check
    node-level laminarity — see `verify_node_laminar` for that, added after
    an external review found this distinction matters: the hyperedge-level
    property holding does not automatically make the node-level assignment
    laminar too, since attach_leftover_nodes had its own (separate, buggy)
    logic."""
    for coarse, fine in zip(levels[:-1], levels[1:]):
        fine_to_coarse = {}
        for c_label, f_label in zip(coarse, fine):
            if f_label in fine_to_coarse and fine_to_coarse[f_label] != c_label:
                return False
            fine_to_coarse[f_label] = c_label
    return True


def verify_node_laminar(assignment: dict) -> tuple[bool, int]:
    """Empirical check on the actual delivered artifact: for every node,
    its coarser-level cluster must be consistent with its finer-level
    cluster (i.e. two nodes sharing a fine cluster must share the same
    coarse cluster too). Returns (all_pass, n_violating_nodes).

    Added after external review found `attach_leftover_nodes` could violate
    this even though the primary `assign_nodes` path (and the hyperedge-
    level check above) did not — P1 is only truly "guaranteed by
    construction" if this passes on the actual output, not just on the
    hyperedge dendrogram."""
    items = [(nid, v) for nid, v in assignment.items() if v[0] is not None]
    if not items:
        return True, 0
    n_levels = len(items[0][1])
    bad_nodes = set()
    for lvl in range(n_levels - 1):
        groups = defaultdict(list)
        for nid, v in items:
            groups[v[lvl + 1]].append((nid, v[lvl]))
        for fine_id, members in groups.items():
            coarse_ids = {c for _, c in members}
            if len(coarse_ids) > 1:
                counts = Counter(c for _, c in members)
                majority = counts.most_common(1)[0][0]
                bad_nodes.update(nid for nid, c in members if c != majority)
    return len(bad_nodes) == 0, len(bad_nodes)


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


def attach_leftover_nodes(nodes: list, all_edges: list, assignment: dict,
                           provenance: dict, n_levels: int) -> dict:
    """Second pass, run only after primary clustering + assignment.

    Some node types (`claim`, `author`, `cited_work` in this dataset) appear
    *only* in the excluded/held-out relation types (`claims`, `authored_by`,
    `cites`), so they never
    get an assignment from `assign_nodes`. Rather than leave them out of the
    hierarchy entirely, attach each such node to the cluster of whichever
    already-assigned node it shares an EXCLUDED_FROM_CLUSTERING or
    HELD_OUT_FOR_COHERENCE hyperedge with (majority vote if more than one
    candidate, at every level, same top-down restriction as the primary
    assignment).

    IMPORTANT (fix for a circularity leak found by external review): a node
    attached via a `cites` edge must NOT be treated as independent evidence
    when `cites` is later used as the T6 coherence probe — that would test
    the clustering against the very relation that placed the node there.
    `provenance[node_id]` is set to the attaching relation ("claims",
    "authored_by" or "cites") for attached nodes
    (or left as "primary" for nodes assigned by `assign_nodes`), and T6 must
    filter out `provenance == "cites"` nodes before running the coherence
    check.
    """
    leftover_edges_by_relation = {
        rel: [e for e in all_edges if e["relation_type"] == rel]
        for rel in (EXCLUDED_FROM_CLUSTERING | HELD_OUT_FOR_COHERENCE)
    }

    unassigned = [n["id"] for n in nodes if assignment.get(n["id"], [None])[0] is None]
    unassigned_set = set(unassigned)

    attached, still_unassigned = 0, 0
    # Process non-probe relations before `cites`: prefer them when a node
    # has several available, to minimise how many nodes end up excluded
    # from the coherence probe. Fix: an earlier version hard-coded
    # ["claims", "cites"] despite this docstring, so `authored_by` was never
    # used and every author node stayed unassigned (318 at 2026).
    for rel in ATTACH_RELATION_ORDER:
        edges_for_rel = leftover_edges_by_relation.get(rel, [])
        neighbor_assignments = defaultdict(list)
        for e in edges_for_rel:
            members = e["members"]
            assigned_members = [m for m in members if m not in unassigned_set and m in assignment]
            for m in members:
                if m in unassigned_set:
                    neighbor_assignments[m].extend(assigned_members)

        for nid in sorted(unassigned_set):  # sorted: deterministic provenance order
            neighbors = neighbor_assignments.get(nid, [])
            if not neighbors:
                continue
            # Top-down restriction (fix for a laminarity bug found by
            # external review): an earlier version computed an
            # INDEPENDENT majority vote per level over the same
            # unrestricted neighbor list, despite a docstring claiming
            # "same top-down restriction as the primary assignment" — it
            # wasn't. That let the majority flip between levels for a
            # node whose neighbors' assignments diverged, producing 10-22
            # non-laminar node paths in practice (all `cites`-provenance,
            # confirmed by an empirical node-level check). This now
            # mirrors assign_nodes exactly: candidate_neighbors shrinks
            # level by level to only those consistent with the
            # already-chosen coarser cluster.
            candidate_neighbors = neighbors
            per_level = []
            for lvl in range(n_levels):
                votes = Counter(assignment[n][lvl] for n in candidate_neighbors if assignment[n][lvl] is not None)
                if not votes:
                    per_level.append(per_level[-1] if per_level else None)
                    continue
                majority_cluster = votes.most_common(1)[0][0]
                per_level.append(majority_cluster)
                candidate_neighbors = [n for n in candidate_neighbors if assignment[n][lvl] == majority_cluster]
            assignment[nid] = per_level
            provenance[nid] = rel
            unassigned_set.discard(nid)
            attached += 1

    still_unassigned = len(unassigned_set)
    print(f"Post-hoc attachment: {attached} nodes attached via held-out/excluded "
          f"relations, {still_unassigned} still fully isolated.")
    return assignment, provenance


def place_articles_via_presents(nodes: list, all_edges: list, node_lookup: dict,
                                 assignment: dict, provenance: dict, n_levels: int) -> None:
    """Deterministically place `article` nodes by inheriting the full level
    path of the method/technique node they present, via `presents` edges
    (arity 2: {article, method}).

    Rationale (from the same review that found the paper-frequency
    weighting fix): `presents` contributes nothing to similarity (its
    arity-2 structure makes its Jaccard overlap with anything else nearly
    always zero) and is excluded from clustering entirely. But articles
    still need *some* principled placement that isn't the generic
    majority-vote post-hoc attachment (which would route them through
    `cites` and reopen the coherence-probe circularity). Since the
    presented method has already been placed by real clustering (it
    appears in `addresses`/`solves`/`uses_*`/`evaluated_on` edges), the
    article can deterministically inherit that placement. Tagged with
    `provenance="presents"` — not `"primary"` (it wasn't clustered) and
    not `"cites"` (it doesn't touch the coherence probe's held-out
    relation), which is exactly the distinction the relaxed probe
    assertion (`provenance != "cites"`) is designed to allow.

    Modifies `assignment` and `provenance` in place.
    """
    presents_edges = [e for e in all_edges if e["relation_type"] == "presents"]
    for e in presents_edges:
        members = e["members"]
        article_ids = [m for m in members if node_lookup.get(m, {}).get("type") == "article"]
        method_ids = [m for m in members if m not in article_ids]
        if not article_ids or not method_ids:
            continue
        article = article_ids[0]
        if assignment.get(article, [None])[0] is not None:
            continue  # already placed (e.g. multiple presents edges; keep first)
        for method in method_ids:
            method_path = assignment.get(method)
            if method_path is not None and method_path[0] is not None:
                assignment[article] = list(method_path)
                provenance[article] = "presents"
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tkh_collection10.json")
    parser.add_argument("--cutoff", type=int, default=None, help="Snapshot cutoff year; omit for full dataset")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--linkage", default="complete", choices=["average", "complete", "ward"],
                         help="complete is the default: average-linkage was found to chain "
                              "into two dominant clusters (75%% of all nodes) on this dataset; "
                              "ward gave similarly balanced results but is only mathematically "
                              "valid for Euclidean distances, which our combined structural+"
                              "semantic distance is not, so it was rejected despite good numbers")
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
    Z = build_dendrogram(dist, method=args.linkage)

    thresholds, levels, k0 = extract_levels(Z, len(edges))
    print(f"Level cluster counts (coarse->fine, target vs achieved): "
          f"targets={list(thresholds)}, achieved={[len(set(l)) for l in levels]}")
    print(f"Level 0 target 10-15, achieved: {k0}")

    laminar_ok = verify_laminar(levels)
    print(f"Laminar refinement check (empirical): {'PASS' if laminar_ok else 'FAIL'}")

    assignment = assign_nodes(nodes, edges, levels)
    provenance = {nid: "primary" for nid, v in assignment.items() if v[0] is not None}
    assigned = len(provenance)
    print(f"Nodes assigned by primary clustering: {assigned}/{len(nodes)} "
          f"({len(nodes) - assigned} have no incident clustering edge)")

    all_edges_for_snapshot = [
        e for e in data["hyperedges"]
        if (args.cutoff is None or (e.get("provenance", {}).get("article_year") is not None
                                     and e["provenance"]["article_year"] <= args.cutoff))
    ]

    place_articles_via_presents(nodes, all_edges_for_snapshot, node_lookup, assignment, provenance, len(levels))
    n_via_presents = sum(1 for p in provenance.values() if p == "presents")
    print(f"Articles placed via presents (deterministic, inherits presented method's cluster): {n_via_presents}")
    assignment, provenance = attach_leftover_nodes(
        nodes, all_edges_for_snapshot, assignment, provenance, len(levels)
    )
    fully_assigned = sum(1 for v in assignment.values() if v[0] is not None)
    cites_attached = sum(1 for p in provenance.values() if p == "cites")
    print(f"Nodes assigned after post-hoc attachment: {fully_assigned}/{len(nodes)} "
          f"({cites_attached} via `cites` — these must be EXCLUDED from the "
          f"T6 coherence probe, which also uses `cites`)")

    node_laminar_ok, n_violations = verify_node_laminar(assignment)
    print(f"Node-level laminar check (empirical, on the actual delivered "
          f"assignment): {'PASS' if node_laminar_ok else f'FAIL ({n_violations} violating nodes)'}")

    Path("outputs").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "cutoff_year": args.cutoff,
            "alpha": args.alpha,
            "linkage": args.linkage,
            "n_nodes": len(nodes),
            "n_clustering_edges": len(edges),
            "level_cluster_counts": [len(set(l)) for l in levels],
            "laminar_check_passed": laminar_ok,
            "node_laminar_check_passed": node_laminar_ok,
            "node_laminar_violations": n_violations,
            "node_assignment": assignment,
            "node_assignment_provenance": provenance,
            "edge_ids": [e["id"] for e in edges],
            "edge_cluster_labels": [[int(x) for x in lvl] for lvl in levels],
        }, f, indent=2, sort_keys=True)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
