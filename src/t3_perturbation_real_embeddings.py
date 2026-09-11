"""
T3 — Corrected perturbation-robustness test.

Fix (found by external review): the previous version perturbed RAW
hyperedges before `merge_evaluated_on_by_article` ran, which could change
which rows get merged into which edge id, producing edge ids the fixed
MiniLM embeddings file was never computed for. That forced an earlier
version onto TF-IDF for this specific test, and even then 13/20
perturbation seeds collapsed to a single level-0 cluster (ARI=0).

Fix: perturb the CLUSTERING-ELIGIBLE edges (AFTER merging), so removing
10% of them never changes any other edge's id — the real, fixed MiniLM
embeddings file can be used directly via subset matching.

Usage:
    python src/t3_perturbation_real_embeddings.py
"""

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


def main():
    data = json.load(open("data/tkh_collection10.json"))
    node_lookup = {n["id"]: n for n in data["nodes"]}
    cutoff = 2026
    nodes, edges = filter_snapshot(data, cutoff)
    all_edges_snap = [e for e in data["hyperedges"]
                       if e.get("provenance", {}).get("article_year") is not None
                       and e["provenance"]["article_year"] <= cutoff]

    print(f"Reference clustering (2026, {len(edges)} clustering-eligible edges, post-merge)...")
    ref_all, ref_clustering_only, ref_k0 = cluster_snapshot(nodes, edges, all_edges_snap, node_lookup)
    print(f"  k0={ref_k0}")

    n_seeds = 100
    aris_all, aris_clustering_only, k0s = [], [], []
    for seed in range(n_seeds):
        perturbed_edges = perturb_clustering_edges(edges, seed=seed, frac=0.10)
        pert_all, pert_clustering_only, pert_k0 = cluster_snapshot(nodes, perturbed_edges, all_edges_snap, node_lookup)
        k0s.append(pert_k0)

        common_all = sorted(set(ref_all) & set(pert_all))
        aris_all.append(adjusted_rand_score([ref_all[n] for n in common_all], [pert_all[n] for n in common_all]))

        common_co = sorted(set(ref_clustering_only) & set(pert_clustering_only))
        aris_clustering_only.append(adjusted_rand_score(
            [ref_clustering_only[n] for n in common_co], [pert_clustering_only[n] for n in common_co]))

        if seed % 20 == 0:
            print(f"  seed={seed}: k0={pert_k0}, ARI(all)={aris_all[-1]:.3f}, "
                  f"ARI(clustering-only)={aris_clustering_only[-1]:.3f}")

    aris_all, aris_clustering_only = np.array(aris_all), np.array(aris_clustering_only)

    def boot_ci(arr):
        rng = np.random.default_rng(0)
        boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(5000)]
        return np.percentile(boot, [2.5, 97.5])

    ci_all = boot_ci(aris_all)
    ci_co = boot_ci(aris_clustering_only)

    print(f"\nPerturbation ARI, {n_seeds} seeds:")
    print(f"  All assigned nodes:       mean={aris_all.mean():.3f}, 95% CI [{ci_all[0]:.3f}, {ci_all[1]:.3f}]")
    print(f"  Clustering-placed only:   mean={aris_clustering_only.mean():.3f}, "
          f"95% CI [{ci_co[0]:.3f}, {ci_co[1]:.3f}]  <- PRIMARY (excludes post-hoc attachment)")
    print(f"k0 range across perturbed seeds: {min(k0s)}-{max(k0s)} "
          f"(all within P2's 10-15 target: {all(10 <= k <= 15 for k in k0s)})")

    hierarchy_paths = {2020: "outputs/hierarchy_2020.json", 2022: "outputs/hierarchy_2022.json",
                       2024: "outputs/hierarchy_2024.json", 2026: "outputs/hierarchy.json"}
    labels_all_by_cutoff, labels_co_by_cutoff = {}, {}
    for c, p in hierarchy_paths.items():
        h = json.load(open(p))
        prov = h.get("node_assignment_provenance", {})
        labels_all_by_cutoff[c] = {nid: v[0] for nid, v in h["node_assignment"].items() if v[0] is not None}
        labels_co_by_cutoff[c] = {nid: l for nid, l in labels_all_by_cutoff[c].items() if prov.get(nid) == "primary"}

    def transition_aris_for(labels_by_cutoff):
        out = []
        for prev_c, curr_c in zip(CUTOFFS[:-1], CUTOFFS[1:]):
            l1, l2 = labels_by_cutoff[prev_c], labels_by_cutoff[curr_c]
            common = sorted(set(l1) & set(l2))
            out.append(adjusted_rand_score([l1[n] for n in common], [l2[n] for n in common]))
        return out

    trans_all = transition_aris_for(labels_all_by_cutoff)
    trans_co = transition_aris_for(labels_co_by_cutoff)

    def per_transition_p(aris, transitions):
        """CORRECTED test (fix: an earlier version compared a single
        transition ARI against the confidence interval of the MEAN
        perturbation ARI — that interval shrinks as seeds are added and
        says little about any one transition. The correct test compares
        each transition against the actual DISTRIBUTION of individual
        seed ARIs: p = fraction of seeds at least as stable as the
        transition.)"""
        return [(1 + (aris >= t).sum()) / (1 + len(aris)) for t in transitions]

    p_all = per_transition_p(aris_all, trans_all)
    p_co = per_transition_p(aris_clustering_only, trans_co)

    print(f"\nCORRECTED per-transition test (p = fraction of {n_seeds} perturbation "
          f"seeds with ARI >= the transition's ARI; low p = transition is more "
          f"stable than typical noise):")
    print(f"  All-nodes transitions:            ARI={[round(a,3) for a in trans_all]}, p={[round(p,3) for p in p_all]}")
    print(f"  Clustering-only transitions (PRIMARY): ARI={[round(a,3) for a in trans_co]}, p={[round(p,3) for p in p_co]}")
    print(f"\nNote: this is NOT an 'equivalent-sized' comparison — perturbation removes "
          f"10% of edges, while real growth adds 40-92% across these transitions. "
          f"Interpret p-values as 'more/less stable than 10%-edge-removal noise', "
          f"not as a matched-magnitude comparison.")

    with open("outputs/t3_perturbation_real_embeddings.json", "w") as f:
        json.dump({
            "method": "perturb post-merge clustering-eligible edges (10% removed), real MiniLM "
                      "embeddings via subset matching; two node sets reported",
            "n_seeds": n_seeds,
            "reference_k0": ref_k0,
            "perturbed_k0_range": [min(k0s), max(k0s)],
            "perturbed_k0_all_within_p2": all(10 <= k <= 15 for k in k0s),
            "perturbation_aris_all_nodes": aris_all.tolist(),
            "perturbation_aris_clustering_only": aris_clustering_only.tolist(),
            "perturbation_ari_mean_all_nodes": float(aris_all.mean()),
            "perturbation_ari_mean_clustering_only": float(aris_clustering_only.mean()),
            "perturbation_ari_ci95_all_nodes": [float(ci_all[0]), float(ci_all[1])],
            "perturbation_ari_ci95_clustering_only": [float(ci_co[0]), float(ci_co[1])],
            "transition_aris_all_nodes": trans_all,
            "transition_aris_clustering_only": trans_co,
            "transition_p_values_all_nodes": p_all,
            "transition_p_values_clustering_only_PRIMARY": p_co,
            "caveat_not_equivalent_sized": "perturbation removes 10% of edges; real growth adds 40-92% "
                                            "across these transitions -- not a matched-magnitude comparison",
        }, f, indent=2)
    print("\nSaved outputs/t3_perturbation_real_embeddings.json")


if __name__ == "__main__":
    main()
