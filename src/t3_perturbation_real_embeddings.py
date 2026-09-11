"""
T3 — Perturbation-robustness test with SNAPSHOT-MATCHED nulls.

Question: is each real cross-snapshot transition (a -> b) at least as
stable as the clustering's own noise floor, i.e. re-clustering after
removing 10% of the clustering-eligible hyperedges?

Fix 1 (earlier external review): perturb the post-merge clustering-eligible
edges, so removing 10% never changes other edge ids and the fixed MiniLM
embeddings file can be used via subset matching.

Fix 2 (this review round): an earlier version compared EVERY transition
against a single null built on the 2026 snapshot. Smaller snapshots cluster
much more stably (mean null ARI ~0.77 at 2020 vs ~0.50 at 2026), so the
2026 null was far too lenient for earlier transitions and produced the
only "significant" result (2022->2024, p=0.03), which also did not survive
multiple-testing correction. Now:
  - a separate 10%-removal null is built for EVERY snapshot;
  - transition a->b is tested against both endpoint nulls, and the
    conservative p = max(p_vs_null(a), p_vs_null(b)) is reported;
  - p-values are Holm-adjusted across the 3 transitions;
  - n_seeds defaults to 1000 so thresholds sit well above the 1/(n+1)
    resolution floor.
The superseded 2026-only p-values are kept in the output, clearly labelled,
for traceability. Caveat unchanged: 10% removal is not a matched-magnitude
comparison to 40-92% real growth.

Usage:
    python src/t3_perturbation_real_embeddings.py [--n-seeds 1000]
"""

import argparse
import contextlib
import io
import json
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "src")
from method import (
    filter_snapshot, structural_similarity_matrix, semantic_similarity_from_embeddings,
    combined_distance_matrix, build_dendrogram, extract_levels, assign_nodes,
    attach_leftover_nodes, place_articles_via_presents,
)

CUTOFFS = [2020, 2022, 2024, 2026]
EMB_PATH = "outputs/semantic_embeddings.npy"
ORDER_PATH = "outputs/embedding_edge_order.json"
ALPHA = 0.5


def cluster_snapshot(nodes, edges, all_edges_snap, node_lookup):
    s_struct = structural_similarity_matrix(edges)
    s_sem = semantic_similarity_from_embeddings(edges, EMB_PATH, ORDER_PATH)
    dist = combined_distance_matrix(s_struct, s_sem, ALPHA)
    Z = build_dendrogram(dist, method="complete")
    thresholds, levels, k0 = extract_levels(Z, len(edges))
    assert 10 <= k0 <= 15, f"k0={k0} outside P2 target"

    assignment = assign_nodes(nodes, edges, levels)
    provenance = {nid: "primary" for nid, v in assignment.items() if v[0] is not None}
    clustering_placed_nodes = set(provenance.keys())  # before post-hoc attachment
    place_articles_via_presents(nodes, all_edges_snap, node_lookup, assignment, provenance, len(levels))
    assignment, provenance = attach_leftover_nodes(nodes, all_edges_snap, assignment, provenance, len(levels))
    node_labels_l0_all = {nid: v[0] for nid, v in assignment.items() if v[0] is not None}
    node_labels_l0_clustering_only = {nid: l for nid, l in node_labels_l0_all.items() if nid in clustering_placed_nodes}
    return node_labels_l0_all, node_labels_l0_clustering_only, k0


def perturb_clustering_edges(edges, seed, frac=0.10):
    rng = np.random.default_rng(seed)
    n_remove = int(round(len(edges) * frac))
    remove_idx = set(rng.choice(len(edges), size=n_remove, replace=False).tolist())
    return [e for i, e in enumerate(edges) if i not in remove_idx]


HIERARCHY_PATHS = {2020: "outputs/hierarchy_2020.json", 2022: "outputs/hierarchy_2022.json",
                   2024: "outputs/hierarchy_2024.json", 2026: "outputs/hierarchy.json"}


def ari_on_common(ref: dict, other: dict) -> float:
    common = sorted(set(ref) & set(other))
    return adjusted_rand_score([ref[n] for n in common], [other[n] for n in common])


def p_upper(null: np.ndarray, observed: float) -> float:
    """p = fraction of null seeds at least as stable as `observed` (+1 smoothing)."""
    return float((1 + (null >= observed).sum()) / (1 + len(null)))


def holm(pvals: list) -> list:
    order = np.argsort(pvals)
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def boot_ci(arr, n_boot=5000):
    rng = np.random.default_rng(0)
    boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def snapshot_null(data, node_lookup, cutoff, n_seeds, shipped):
    nodes, edges = filter_snapshot(data, cutoff)
    all_edges_snap = [e for e in data["hyperedges"]
                      if e.get("provenance", {}).get("article_year") is not None
                      and e["provenance"]["article_year"] <= cutoff]
    with contextlib.redirect_stdout(io.StringIO()):  # silence per-run attachment logs
        ref_all, ref_co, ref_k0 = cluster_snapshot(nodes, edges, all_edges_snap, node_lookup)
        # The reference must BE the shipped hierarchy, or the null is about a different object.
        if ref_all != shipped["all"] or ref_co != shipped["co"]:
            raise RuntimeError(f"{cutoff}: re-clustering differs from shipped hierarchy; re-run method.py")
        aris_all, aris_co, k0s = [], [], []
        for seed in range(n_seeds):
            pa, pc, k0 = cluster_snapshot(nodes, perturb_clustering_edges(edges, seed), all_edges_snap, node_lookup)
            aris_all.append(ari_on_common(ref_all, pa))
            aris_co.append(ari_on_common(ref_co, pc))
            k0s.append(k0)
    aris_all, aris_co = np.array(aris_all), np.array(aris_co)
    print(f"  {cutoff}: {len(edges)} edges, ref k0={ref_k0}; null ARI clustering-only "
          f"mean={aris_co.mean():.3f}, all-nodes mean={aris_all.mean():.3f}; k0 range {min(k0s)}-{max(k0s)}")
    return {
        "n_clustering_edges": len(edges), "reference_k0": ref_k0,
        "perturbed_k0_range": [min(k0s), max(k0s)],
        "perturbed_k0_all_within_p2": all(10 <= k <= 15 for k in k0s),
        "aris_clustering_only": aris_co, "aris_all_nodes": aris_all,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=1000)
    ap.add_argument("--out", default="outputs/t3_perturbation_real_embeddings.json")
    args = ap.parse_args()

    data = json.load(open("data/tkh_collection10.json"))
    node_lookup = {n["id"]: n for n in data["nodes"]}

    shipped = {}
    for c, path in HIERARCHY_PATHS.items():
        h = json.load(open(path))
        prov = h["node_assignment_provenance"]
        lab_all = {nid: v[0] for nid, v in h["node_assignment"].items() if v[0] is not None}
        shipped[c] = {"all": lab_all, "co": {n: l for n, l in lab_all.items() if prov.get(n) == "primary"}}

    print(f"Building 10%-removal nulls, {args.n_seeds} seeds per snapshot...")
    nulls = {c: snapshot_null(data, node_lookup, c, args.n_seeds, shipped[c]) for c in CUTOFFS}

    transitions, rows = list(zip(CUTOFFS[:-1], CUTOFFS[1:])), {"clustering_only": [], "all_nodes": []}
    for key, lab_key, null_key in [("clustering_only", "co", "aris_clustering_only"),
                                   ("all_nodes", "all", "aris_all_nodes")]:
        for a, b in transitions:
            obs = ari_on_common(shipped[a][lab_key], shipped[b][lab_key])
            p_prev, p_curr = p_upper(nulls[a][null_key], obs), p_upper(nulls[b][null_key], obs)
            rows[key].append({
                "transition": f"{a}->{b}", "observed_ari": obs,
                "p_vs_prev_snapshot_null": p_prev, "p_vs_curr_snapshot_null": p_curr,
                "p_conservative": max(p_prev, p_curr),
                "p_vs_2026_null_SUPERSEDED": p_upper(nulls[2026][null_key], obs),
            })
        for r, adj in zip(rows[key], holm([r["p_conservative"] for r in rows[key]])):
            r["p_conservative_holm"] = adj

    for key in rows:
        print(f"\n{key}{' (PRIMARY)' if key == 'clustering_only' else ''}:")
        for r in rows[key]:
            print(f"  {r['transition']}: ARI={r['observed_ari']:.3f}  p_prev={r['p_vs_prev_snapshot_null']:.3f}  "
                  f"p_curr={r['p_vs_curr_snapshot_null']:.3f}  p_cons(Holm)={r['p_conservative_holm']:.3f}  "
                  f"[superseded 2026-null p={r['p_vs_2026_null_SUPERSEDED']:.3f}]")
    n_sig = sum(r["p_conservative_holm"] < 0.05 for r in rows["clustering_only"])

    out = {
        "method": "per-snapshot null: re-cluster after removing 10% of post-merge clustering-eligible "
                  "edges (real MiniLM embeddings via subset matching); each transition tested against "
                  "both endpoint nulls, conservative p = max, Holm-adjusted across transitions",
        "n_seeds": args.n_seeds,
        "null_by_snapshot": {
            str(c): {
                "n_clustering_edges": v["n_clustering_edges"], "reference_k0": v["reference_k0"],
                "perturbed_k0_range": v["perturbed_k0_range"],
                "perturbed_k0_all_within_p2": v["perturbed_k0_all_within_p2"],
                "ari_mean_clustering_only": float(v["aris_clustering_only"].mean()),
                "ari_ci95_of_mean_clustering_only": boot_ci(v["aris_clustering_only"]),
                "ari_2.5_97.5_pct_clustering_only": [float(x) for x in np.percentile(v["aris_clustering_only"], [2.5, 97.5])],
                "ari_mean_all_nodes": float(v["aris_all_nodes"].mean()),
                "ari_ci95_of_mean_all_nodes": boot_ci(v["aris_all_nodes"]),
                "aris_clustering_only": v["aris_clustering_only"].tolist(),
                "aris_all_nodes": v["aris_all_nodes"].tolist(),
            } for c, v in nulls.items()
        },
        "transitions_clustering_only_PRIMARY": rows["clustering_only"],
        "transitions_all_nodes_secondary": rows["all_nodes"],
        "n_transitions_significant_primary_holm_0.05": n_sig,
        "interpretation": (f"{n_sig}/3 transitions are more stable than their own snapshots' 10%-edge-removal "
                           "noise after Holm correction. The earlier claim that 2022->2024 is robustly stable "
                           "(p=0.03) came from comparing against the 2026 null, which is much less stable than "
                           "the 2022/2024 nulls, and is withdrawn."),
        "caveat_not_equivalent_sized": "perturbation removes 10% of edges; real growth adds 40-92% across "
                                       "these transitions -- not a matched-magnitude comparison",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
