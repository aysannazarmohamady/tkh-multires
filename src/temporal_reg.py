"""
T3 prep — temporal regularization on the hyperedge distance matrix, and the
decisive perturbation-robustness test for whether it's real stabilization
or just path-dependent lock-in.

Mechanism: for a snapshot t (t > 2020), before running HAC, reduce the
distance between any two hyperedges (i, j) that BOTH existed in snapshot
t-1 AND were in the same level-0 cluster at t-1:
    d'(i,j) = clip(d(i,j) - lambda, 0, None)
New edges (not present at t-1) are untouched. lambda=0 recovers the
original (non-regularized) method exactly.

Decisive test (requested): does lambda=0.2 improve ARI under perturbation
(10% of all hyperedges removed, 5 seeds), or does it just lock in whatever
the 2020 clustering happened to be? If perturbed-vs-reference ARI at
lambda=0.2 is meaningfully higher than at lambda=0, the mechanism is really
stabilizing against noise, not just repeating a fixed 2020 decision.
"""

import json
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "src")
from method import (
    filter_snapshot, structural_similarity_matrix, semantic_similarity_from_embeddings,
    build_dendrogram, extract_levels, assign_nodes, attach_leftover_nodes,
)

CUTOFFS = [2020, 2022, 2024, 2026]


def apply_temporal_regularization(dist, edges, prev_edge_labels, lam):
    """prev_edge_labels: {edge_id: level0_cluster_label} from the PREVIOUS
    snapshot in the chain, or None for the first snapshot (2020)."""
    if not prev_edge_labels or lam == 0:
        return dist
    ids = [e["id"] for e in edges]
    m = len(ids)
    same_cluster_prev = np.zeros((m, m), dtype=bool)
    for i in range(m):
        li = prev_edge_labels.get(ids[i])
        if li is None:
            continue
        for j in range(i + 1, m):
            lj = prev_edge_labels.get(ids[j])
            if lj is not None and li == lj:
                same_cluster_prev[i, j] = True
                same_cluster_prev[j, i] = True
    reg = dist - lam * same_cluster_prev
    return np.clip(reg, 0, None)


def run_chain(data, alpha, lam, embeddings_path, order_path, cutoffs=CUTOFFS):
    """Runs the full 2020->2022->2024->2026 chain with temporal
    regularization. Returns {cutoff: {"node_labels_l0": {...}, "edge_labels_l0": {...}}}."""
    results = {}
    prev_edge_labels = None
    for cutoff in cutoffs:
        nodes, edges = filter_snapshot(data, cutoff)
        s_struct = structural_similarity_matrix(edges)
        s_sem = semantic_similarity_from_embeddings(edges, embeddings_path, order_path)
        sim = alpha * s_struct + (1 - alpha) * s_sem
        dist = 1 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.clip((dist + dist.T) / 2, 0, None)

        dist = apply_temporal_regularization(dist, edges, prev_edge_labels, lam)

        Z = build_dendrogram(dist, method="complete")
        thresholds, levels, k0 = extract_levels(Z, len(edges))

        edge_ids = [e["id"] for e in edges]
        edge_labels_l0 = {eid: int(lbl) for eid, lbl in zip(edge_ids, levels[0])}

        assignment = assign_nodes(nodes, edges, levels)
        provenance = {nid: "primary" for nid, v in assignment.items() if v[0] is not None}
        all_edges_snap = [e for e in data["hyperedges"]
                           if e.get("year") is not None and e["year"] <= cutoff]
        assignment, provenance = attach_leftover_nodes(nodes, all_edges_snap, assignment, provenance, len(levels))

        node_labels_l0 = {nid: v[0] for nid, v in assignment.items() if v[0] is not None}
        results[cutoff] = {"node_labels_l0": node_labels_l0, "edge_labels_l0": edge_labels_l0,
                            "k0": k0}
        prev_edge_labels = edge_labels_l0
    return results


def perturb_data(data, seed, frac=0.10):
    rng = np.random.default_rng(seed)
    edges = data["hyperedges"]
    n_remove = int(round(len(edges) * frac))
    remove_idx = set(rng.choice(len(edges), size=n_remove, replace=False).tolist())
    kept_edges = [e for i, e in enumerate(edges) if i not in remove_idx]
    return {"nodes": data["nodes"], "hyperedges": kept_edges}


def main():
    data = json.load(open("data/tkh_collection10.json"))
    emb_path = "outputs/semantic_embeddings.npy"
    order_path = "outputs/embedding_edge_order.json"
    alpha = 0.5

    print("Running reference chains (unperturbed) for lambda=0 and lambda=0.2...")
    ref = {}
    for lam in [0.0, 0.2]:
        ref[lam] = run_chain(data, alpha, lam, emb_path, order_path)
        print(f"  lambda={lam}: k0 per snapshot = {[ref[lam][c]['k0'] for c in CUTOFFS]}")

    n_seeds = 5
    print(f"\nRunning {n_seeds} perturbation seeds (remove 10% of all hyperedges) "
          f"for lambda in [0.0, 0.2]...")

    ari_by_lambda = {0.0: [], 0.2: []}
    for lam in [0.0, 0.2]:
        for seed in range(n_seeds):
            pdata = perturb_data(data, seed=seed, frac=0.10)
            perturbed = run_chain(pdata, alpha, lam, emb_path, order_path)
            ref_labels = ref[lam][2026]["node_labels_l0"]
            pert_labels = perturbed[2026]["node_labels_l0"]
            common = sorted(set(ref_labels) & set(pert_labels))
            a = [ref_labels[n] for n in common]
            b = [pert_labels[n] for n in common]
            ari = adjusted_rand_score(a, b)
            ari_by_lambda[lam].append(ari)
            print(f"  lambda={lam} seed={seed}: n_common={len(common)}, ARI={ari:.3f}")

    print("\n--- Decisive perturbation-robustness result ---")
    for lam in [0.0, 0.2]:
        arr = np.array(ari_by_lambda[lam])
        print(f"lambda={lam}: mean ARI = {arr.mean():.3f} +/- {arr.std():.3f} "
              f"(n={len(arr)} seeds)")

    with open("outputs/perturbation_robustness_check.json", "w") as f:
        json.dump({
            "alpha": alpha,
            "n_seeds": n_seeds,
            "perturb_frac": 0.10,
            "ari_by_lambda": {str(k): v for k, v in ari_by_lambda.items()},
            "mean_ari_by_lambda": {str(k): float(np.mean(v)) for k, v in ari_by_lambda.items()},
            "std_ari_by_lambda": {str(k): float(np.std(v)) for k, v in ari_by_lambda.items()},
        }, f, indent=2)
    print("\nSaved outputs/perturbation_robustness_check.json")


if __name__ == "__main__":
    main()
