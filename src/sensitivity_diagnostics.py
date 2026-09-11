"""
Quantifies two documented sensitivities of src/method.py WITHOUT changing
the shipped method (changing either re-clusters every snapshot and
invalidates all labels; see report.md, "Limitations").

(1) Rank normalisation of the structural signal. 92-95% of structural
    pairs are exactly zero; `rankdata(method="average")` gives all of them
    one tied mid-rank (~0.47), while the smallest non-zero value ranks
    ~0.92+. Structure therefore acts mostly as a binary "any overlap" bonus.
    Reported per snapshot: zero fraction, tied-zero rank, smallest non-zero
    rank, structural variance share at alpha=0.5 for the shipped
    normalisation and for a zero-pinned alternative (zeros -> 0, non-zeros
    ranked among themselves), and node-level ARI (primary-assigned nodes,
    levels 0-2) between the two clusterings, next to the snapshot's own
    10%-removal noise band from outputs/t3_perturbation_real_embeddings.json.

    Variance share := var(a*r_struct) / (var(a*r_struct) + var((1-a)*r_sem))
    over upper-triangle pairs (the covariance term is excluded).

(2) Majority-vote tie-breaking in assign_nodes. Counter.most_common breaks
    ties by first insertion, i.e. edge-list order. Reported: share of
    primary-assigned nodes that meet a tie at any level; nodes whose path
    changes when the edge order is permuted (cluster labels held fixed, so
    ONLY the tie-break can change anything); node-level laminarity across
    all permutations; and how many existing labelled clusters would change
    n_members under a canonical tie-break (smallest cluster id wins).

Usage:
    python src/sensitivity_diagnostics.py   # writes outputs/sensitivity_diagnostics.json
"""

import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "src")
from method import (  # noqa: E402
    assign_nodes, build_dendrogram, extract_levels, filter_snapshot,
    semantic_similarity_from_embeddings, structural_similarity_matrix, verify_node_laminar,
)

CUTOFFS = [2020, 2022, 2024, 2026]
HIERARCHY_PATHS = {2020: "outputs/hierarchy_2020.json", 2022: "outputs/hierarchy_2022.json",
                   2024: "outputs/hierarchy_2024.json", 2026: "outputs/hierarchy.json"}
EMB, ORDER = "outputs/semantic_embeddings.npy", "outputs/embedding_edge_order.json"
ALPHA, N_PERM = 0.5, 20
NON_TOPICAL = ("claim", "cited_work", "author")


def upper(m):
    return m[np.triu_indices_from(m, k=1)]


def rank_average(v):  # shipped normalisation
    return rankdata(v, method="average") / len(v)


def rank_zero_pinned(v):  # alternative: zeros stay 0, non-zeros ranked among themselves
    r, nz = np.zeros_like(v, dtype=float), v > 0
    if nz.any():
        r[nz] = rankdata(v[nz], method="average") / nz.sum()
    return r


def to_matrix(vals, n):
    out = np.zeros((n, n))
    out[np.triu_indices(n, k=1)] = vals
    out = out + out.T
    np.fill_diagonal(out, 1.0)
    return out


def cluster_levels(s_struct, s_sem, rank_fn):
    n = len(s_struct)
    rs, rm = rank_fn(upper(s_struct)), rank_average(upper(s_sem))
    dist = 1 - to_matrix(ALPHA * rs + (1 - ALPHA) * rm, n)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    _, levels, _ = extract_levels(build_dendrogram(dist, method="complete"), n)
    share = np.var(ALPHA * rs) / (np.var(ALPHA * rs) + np.var((1 - ALPHA) * rm))
    return levels, float(share)


def has_tie(nid, node_edges, levels):
    cand = node_edges.get(nid, [])
    for labels in levels:
        counts = Counter(labels[e] for e in cand).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            return True
        cand = [e for e in cand if labels[e] == counts[0][0]]
    return False


def canonical_assign(nodes, edges, levels):
    """assign_nodes with a canonical tie-break (smallest cluster id among the maxima)."""
    inc = defaultdict(list)
    for i, e in enumerate(edges):
        for m in e["members"]:
            inc[m].append(i)
    out = {}
    for n in nodes:
        cand = inc.get(n["id"], [])
        if not cand:
            continue
        path = []
        for labels in levels:
            counts = Counter(int(labels[e]) for e in cand)
            best = min(counts, key=lambda c: (-counts[c], c))
            path.append(best)
            cand = [e for e in cand if labels[e] == best]
        out[n["id"]] = path
    return out


def load_label_counts():
    """{(year, level): {cluster_id: n_members}} from the labels file (all blocks)."""
    raw = json.load(open("outputs/labelling_faithfulness_pilot.json"))
    blocks = {(2026, 0): ["clusters"], (2020, 0): ["clusters_2020_level0"],
              (2022, 0): ["clusters_2022_level0", "clusters_2022_level0_continued"],
              (2024, 0): ["clusters_2024_level0", "clusters_2024_level0_continued"]}
    out = defaultdict(dict)
    for key, names in blocks.items():
        for name in names:
            for e in raw.get(name, []):
                out[key][int(e["cluster_id"])] = e["n_members"]
    for y in CUTOFFS:
        for cid, e in raw.get(f"level1_{y}", {}).get("clusters", {}).items():
            out[(y, 1)][int(cid)] = e["n_members"]
    return out


def main():
    data = json.load(open("data/tkh_collection10.json"))
    node_type = {n["id"]: n["type"] for n in data["nodes"]}
    t3 = json.load(open("outputs/t3_perturbation_real_embeddings.json"))["null_by_snapshot"]
    label_counts = load_label_counts()
    rng = np.random.default_rng(0)
    result = {"alpha": ALPHA, "n_edge_order_permutations": N_PERM, "per_snapshot": {}}
    n_label_changed = n_labelled = 0

    for c in CUTOFFS:
        hier = json.load(open(HIERARCHY_PATHS[c]))
        nodes, edges = filter_snapshot(data, c)
        assert [e["id"] for e in edges] == hier["edge_ids"], f"{c}: edge order differs from shipped hierarchy"
        s_struct = structural_similarity_matrix(edges)
        s_sem = semantic_similarity_from_embeddings(edges, EMB, ORDER)
        vs = upper(s_struct)
        r_avg = rank_average(vs)
        zero = vs == 0

        lv_ship, share_ship = cluster_levels(s_struct, s_sem, rank_average)
        assert [list(map(int, l)) for l in lv_ship] == hier["edge_cluster_labels"], f"{c}: not the shipped clustering"
        lv_pin, share_pin = cluster_levels(s_struct, s_sem, rank_zero_pinned)
        a_ship, a_pin = assign_nodes(nodes, edges, lv_ship), assign_nodes(nodes, edges, lv_pin)
        primary = sorted(n for n, v in a_ship.items() if v[0] is not None)
        ari_by_level = [adjusted_rand_score([a_ship[n][l] for n in primary], [a_pin[n][l] for n in primary])
                        for l in range(3)]

        # (2) tie-breaking
        node_edges = defaultdict(list)
        for i, e in enumerate(edges):
            for m in e["members"]:
                node_edges[m].append(i)
        tied = {n for n in primary if has_tie(n, node_edges, lv_ship)}
        flipped_counts, flipped_union, laminar_all = [], set(), True
        for _ in range(N_PERM):
            perm = rng.permutation(len(edges))
            a_perm = assign_nodes(nodes, [edges[i] for i in perm], [l[perm] for l in lv_ship])
            flipped = {n for n in primary if a_perm[n] != a_ship[n]}
            flipped_counts.append(len(flipped))
            flipped_union |= flipped
            laminar_all &= verify_node_laminar({n: a_perm[n] for n in primary})[0]

        canon = canonical_assign(nodes, edges, lv_ship)
        for lvl in (0, 1):
            counts = Counter(p[lvl] for n, p in canon.items() if node_type.get(n) not in NON_TOPICAL)
            for cid, n_mem in label_counts.get((c, lvl), {}).items():
                n_labelled += 1
                n_label_changed += counts.get(cid, 0) != n_mem

        noise = t3[str(c)]["ari_2.5_97.5_pct_clustering_only"]
        result["per_snapshot"][str(c)] = {
            "structural_zero_fraction": float(zero.mean()),
            "tied_zero_rank": float(r_avg[zero][0]) if zero.any() else None,
            "smallest_nonzero_rank": float(r_avg[~zero].min()) if (~zero).any() else None,
            "structural_variance_share_shipped": share_ship,
            "structural_variance_share_zero_pinned": share_pin,
            "ari_shipped_vs_zero_pinned_levels_0_2": ari_by_level,
            "own_10pct_removal_noise_band_L0_2.5_97.5pct": noise,
            "n_primary_nodes": len(primary),
            "nodes_with_tie_on_path": len(tied),
            "tie_share_of_primary": len(tied) / len(primary),
            "flipped_under_permutation_mean": float(np.mean(flipped_counts)),
            "flipped_union_subset_of_tied": flipped_union <= tied,
            "node_laminar_all_permutations": bool(laminar_all),
        }
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result["per_snapshot"][str(c)].items()})

    result["canonical_tiebreak_changes_labelled_n_members"] = {"changed": n_label_changed, "labelled": n_labelled}
    print("labelled clusters whose n_members would change under canonical tie-break:", n_label_changed, "/", n_labelled)
    with open("outputs/sensitivity_diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved outputs/sensitivity_diagnostics.json")


if __name__ == "__main__":
    main()
