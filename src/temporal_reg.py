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
    combined_distance_matrix, build_dendrogram, extract_levels, assign_nodes, attach_leftover_nodes,
)

CUTOFFS = [2020, 2022, 2024, 2026]


def apply_temporal_regularization(dist, edges, prev_edge_labels, lam_sd, shuffle_seed=None):
    """prev_edge_labels: {edge_id: level0_cluster_label} from the PREVIOUS
    snapshot in the chain, or None for the first snapshot (2020).

    lam_sd: lambda expressed in units of dist's own upper-triangle std
    (fix for a scale bug found by external review: a raw lambda=0.2 against
    this distance distribution's actual std (~0.081, post rank-
    normalization it differs — recomputed fresh here) turned out to be
    ~2.5 std, which clips historical pairs to near-zero distance and
    re-imposes the previous dendrogram almost verbatim, rather than acting
    as a soft prior).

    shuffle_seed: if not None, prev_edge_labels' VALUES are randomly
    permuted across edge ids before use — this is the null model requested
    to check whether "stability" gains are just circularity (the metric
    directly manipulated) rather than a real effect. None = real prior.
    """
    if not prev_edge_labels or lam_sd == 0:
        return dist
    ids = [e["id"] for e in edges]
    iu = np.triu_indices_from(dist, k=1)
    sd = dist[iu].std()
    lam = lam_sd * sd

    labels_to_use = prev_edge_labels
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        keys = list(prev_edge_labels.keys())
        vals = list(prev_edge_labels.values())
        rng.shuffle(vals)
        labels_to_use = dict(zip(keys, vals))

    m = len(ids)
    same_cluster_prev = np.zeros((m, m), dtype=bool)
    for i in range(m):
        li = labels_to_use.get(ids[i])
        if li is None:
            continue
        for j in range(i + 1, m):
            lj = labels_to_use.get(ids[j])
            if lj is not None and li == lj:
                same_cluster_prev[i, j] = True
                same_cluster_prev[j, i] = True
    reg = dist - lam * same_cluster_prev
    return np.clip(reg, 0, None)


def run_chain(data, alpha, lam_sd, embeddings_path, order_path, cutoffs=CUTOFFS, shuffle_null=False):
    """Runs the full 2020->2022->2024->2026 chain with temporal
    regularization. Uses the ACTUAL rank-normalized combined_distance_matrix
    (fix: an earlier version recomputed `sim` manually on raw matrices,
    bypassing the rank-normalization fix applied to method.py — every prior
    T3 artifact was built on the pre-fix, ~95%-semantic raw mixture).
    Returns {cutoff: {"node_labels_l0": {...}, "edge_labels_l0": {...}, "k0": int}}."""
    results = {}
    prev_edge_labels = None
    for idx, cutoff in enumerate(cutoffs):
        nodes, edges = filter_snapshot(data, cutoff)
        s_struct = structural_similarity_matrix(edges)
        s_sem = semantic_similarity_from_embeddings(edges, embeddings_path, order_path)
        dist = combined_distance_matrix(s_struct, s_sem, alpha)

        seed = idx if shuffle_null else None  # distinct shuffle per snapshot transition
        dist = apply_temporal_regularization(dist, edges, prev_edge_labels, lam_sd, shuffle_seed=seed)

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


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    data = json.load(open("data/tkh_collection10.json"))
    emb_path = "outputs/semantic_embeddings.npy"
    order_path = "outputs/embedding_edge_order.json"
    alpha = 0.5
    lam_sd_grid = [0.0, 0.1, 0.25, 0.5]
    n_seeds = 5

    print("=== Step 1: lambda sweep (in units of distance std), selected by PERTURBATION-ARI ===")
    ref_by_lam = {}
    perturbation_results = {}
    for lam_sd in lam_sd_grid:
        ref_by_lam[lam_sd] = run_chain(data, alpha, lam_sd, emb_path, order_path)
        aris = []
        for seed in range(n_seeds):
            pdata = perturb_data(data, seed=seed, frac=0.10)
            perturbed = run_chain(pdata, alpha, lam_sd, emb_path, order_path)
            ref_labels = ref_by_lam[lam_sd][2026]["node_labels_l0"]
            pert_labels = perturbed[2026]["node_labels_l0"]
            common = sorted(set(ref_labels) & set(pert_labels))
            a = [ref_labels[n] for n in common]
            b = [pert_labels[n] for n in common]
            aris.append(adjusted_rand_score(a, b))
        perturbation_results[lam_sd] = aris
        print(f"  lambda_sd={lam_sd}: perturbation ARI = {np.mean(aris):.3f} +/- {np.std(aris):.3f} "
              f"(k0 per snapshot = {[ref_by_lam[lam_sd][c]['k0'] for c in CUTOFFS]})")

    best_lam_sd = max(lam_sd_grid, key=lambda l: np.mean(perturbation_results[l]))
    print(f"\nSelected lambda_sd = {best_lam_sd} (highest mean perturbation-ARI; "
          f"transition-ARI was NOT used for this decision, since the regularizer "
          f"directly manipulates transition-ARI's own metric — see report.md).")

    print(f"\n=== Step 2: null model (shuffled previous-snapshot labels) at "
          f"lambda_sd={best_lam_sd} ===")
    from sklearn.metrics import adjusted_rand_score as ari_fn

    def transition_aris(chain):
        out = []
        for prev_c, curr_c in zip(CUTOFFS[:-1], CUTOFFS[1:]):
            l1 = chain[prev_c]["node_labels_l0"]
            l2 = chain[curr_c]["node_labels_l0"]
            common = sorted(set(l1) & set(l2))
            out.append(ari_fn([l1[n] for n in common], [l2[n] for n in common]))
        return out

    observed_chain = ref_by_lam[best_lam_sd]
    observed_transition_aris = transition_aris(observed_chain)

    null_transition_aris = []
    for seed in range(n_seeds):
        null_chain = run_chain(data, alpha, best_lam_sd, emb_path, order_path, shuffle_null=True)
        null_transition_aris.append(transition_aris(null_chain))
    null_transition_aris = np.array(null_transition_aris)  # shape (n_seeds, n_transitions)

    print(f"Observed transition ARI (lambda_sd={best_lam_sd}): "
          f"{[round(a, 3) for a in observed_transition_aris]}")
    print(f"Null (shuffled prior) transition ARI, mean per transition: "
          f"{[round(x, 3) for x in null_transition_aris.mean(axis=0)]}")

    obs_lo, obs_hi = bootstrap_ci(observed_transition_aris)
    null_lo, null_hi = bootstrap_ci(null_transition_aris.mean(axis=1))
    print(f"Observed mean transition ARI: {np.mean(observed_transition_aris):.3f} "
          f"(95% CI [{obs_lo:.3f}, {obs_hi:.3f}])")
    print(f"Null mean transition ARI: {null_transition_aris.mean():.3f} "
          f"(95% CI [{null_lo:.3f}, {null_hi:.3f}])")

    result = {
        "alpha": alpha,
        "lambda_sd_grid": lam_sd_grid,
        "perturbation_ari_by_lambda_sd": {str(k): v for k, v in perturbation_results.items()},
        "mean_perturbation_ari_by_lambda_sd": {str(k): float(np.mean(v)) for k, v in perturbation_results.items()},
        "selected_lambda_sd": best_lam_sd,
        "selection_criterion": "max mean perturbation-ARI (NOT transition-ARI, "
                                "since the regularizer directly manipulates transition-ARI)",
        "observed_transition_aris": [float(a) for a in observed_transition_aris],
        "observed_transition_ari_mean": float(np.mean(observed_transition_aris)),
        "observed_transition_ari_bootstrap_ci95": [obs_lo, obs_hi],
        "null_transition_aris_per_seed": null_transition_aris.tolist(),
        "null_transition_ari_mean": float(null_transition_aris.mean()),
        "null_transition_ari_bootstrap_ci95": [null_lo, null_hi],
    }
    with open("outputs/lambda_selection_and_null.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved outputs/lambda_selection_and_null.json")
    return best_lam_sd


if __name__ == "__main__":
    main()
