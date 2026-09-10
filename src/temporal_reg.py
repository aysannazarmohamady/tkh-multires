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
    filter_snapshot, structural_similarity_matrix, semantic_similarity_matrix,
    combined_distance_matrix, build_dendrogram, extract_levels, assign_nodes,
    attach_leftover_nodes, place_articles_via_presents,
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
    bypassing the rank-normalization fix applied to method.py).

    Uses TF-IDF (not the fixed sentence-embeddings file) for the semantic
    signal: this module's perturbation tests remove random raw hyperedges,
    which can change which rows get merged by
    `merge_evaluated_on_by_article` and produce edge ids the fixed
    embeddings file was never computed for. TF-IDF has no such fixed-
    vocabulary dependency. `embeddings_path`/`order_path` are accepted for
    interface compatibility but unused; kept so callers don't need to
    change.
    """
    results = {}
    prev_edge_labels = None
    for idx, cutoff in enumerate(cutoffs):
        nodes, edges = filter_snapshot(data, cutoff)
        node_lookup = {n["id"]: n for n in data["nodes"]}
        s_struct = structural_similarity_matrix(edges)
        s_sem = semantic_similarity_matrix(edges, node_lookup)
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
                           if e.get("provenance", {}).get("article_year") is not None
                           and e["provenance"]["article_year"] <= cutoff]
        place_articles_via_presents(nodes, all_edges_snap, node_lookup, assignment, provenance, len(levels))
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
    removed_edges = [e for i, e in enumerate(edges) if i in remove_idx]
    return {"nodes": data["nodes"], "hyperedges": kept_edges}, removed_edges


def nodes_touched_by_edges(edges: list) -> set:
    touched = set()
    for e in edges:
        touched.update(e["members"])
    return touched


def ari_changed_vs_unchanged(ref_labels: dict, pert_labels: dict, touched_nodes: set):
    """Split ARI by whether a node's incident (removed) edges were touched
    by the perturbation, per external review's request: stability should
    be visibly higher for untouched nodes if the method is doing anything
    beyond noise. Returns (ari_unchanged, ari_changed, n_unchanged, n_changed)."""
    common = sorted(set(ref_labels) & set(pert_labels))
    unchanged = [n for n in common if n not in touched_nodes]
    changed = [n for n in common if n in touched_nodes]

    def ari_of(subset):
        if len(subset) < 2:
            return None
        a = [ref_labels[n] for n in subset]
        b = [pert_labels[n] for n in subset]
        return adjusted_rand_score(a, b)

    return ari_of(unchanged), ari_of(changed), len(unchanged), len(changed)


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def shuffle_new_edges_arity_preserving(prev_data, curr_data, seed):
    """Structure-preservation null for cross-snapshot transitions (fix:
    finding that perturbation-ARI (~0.46) and real transition-ARI
    (~0.43-0.53) were indistinguishable, so real corpus growth looked no
    more "structure-preserving" than noise). Build a synthetic version of
    `curr_data` where every edge NEW at curr (present in curr, absent from
    prev, matched by id) has its membership randomly reassigned among
    nodes present at curr, preserving each edge's original arity exactly.
    Old edges (present in both) are untouched. If real growth is more
    structure-preserving than random growth, the real transition ARI
    should be measurably higher than ARI against this synthetic version.
    """
    rng = np.random.default_rng(seed)
    prev_ids = {e["id"] for e in prev_data["hyperedges"]}
    curr_node_ids = [n["id"] for n in curr_data["nodes"]]
    new_edges, old_edges = [], []
    for e in curr_data["hyperedges"]:
        (new_edges if e["id"] not in prev_ids else old_edges).append(e)

    shuffled_new = []
    for e in new_edges:
        arity = len(e["members"])
        new_members = list(rng.choice(curr_node_ids, size=arity, replace=False))
        shuffled_new.append({**e, "members": new_members})

    return {"nodes": curr_data["nodes"], "hyperedges": old_edges + shuffled_new}


def main():
    data = json.load(open("data/tkh_collection10.json"))
    emb_path = "outputs/semantic_embeddings.npy"
    order_path = "outputs/embedding_edge_order.json"
    alpha = 0.5
    lam_sd_grid = [0.0, 0.1, 0.25, 0.5]
    n_seeds_screen = 5
    n_seeds_final = 20

    print("=== Step 1: lambda sweep (5-seed screen), selected by PERTURBATION-ARI ===")
    ref_by_lam = {}
    perturbation_results = {}
    for lam_sd in lam_sd_grid:
        ref_by_lam[lam_sd] = run_chain(data, alpha, lam_sd, emb_path, order_path)
        aris = []
        for seed in range(n_seeds_screen):
            pdata, _ = perturb_data(data, seed=seed, frac=0.10)
            perturbed = run_chain(pdata, alpha, lam_sd, emb_path, order_path)
            ref_labels = ref_by_lam[lam_sd][2026]["node_labels_l0"]
            pert_labels = perturbed[2026]["node_labels_l0"]
            common = sorted(set(ref_labels) & set(pert_labels))
            aris.append(adjusted_rand_score([ref_labels[n] for n in common], [pert_labels[n] for n in common]))
        perturbation_results[lam_sd] = aris
        print(f"  lambda_sd={lam_sd}: perturbation ARI = {np.mean(aris):.3f} +/- {np.std(aris):.3f}")

    best_lam_sd = max(lam_sd_grid, key=lambda l: np.mean(perturbation_results[l]))
    print(f"\nSelected lambda_sd = {best_lam_sd} (highest mean perturbation-ARI).")

    print(f"\n=== Step 2: {n_seeds_final}-seed final validation at lambda_sd={best_lam_sd}, "
          f"with changed-vs-unchanged node breakdown ===")
    ref_labels_final = ref_by_lam[best_lam_sd][2026]["node_labels_l0"]
    overall_aris, unchanged_aris, changed_aris = [], [], []
    n_unchanged_list, n_changed_list = [], []
    for seed in range(n_seeds_final):
        pdata, removed_edges = perturb_data(data, seed=seed, frac=0.10)
        touched = nodes_touched_by_edges(removed_edges)
        perturbed = run_chain(pdata, alpha, best_lam_sd, emb_path, order_path)
        pert_labels = perturbed[2026]["node_labels_l0"]
        common = sorted(set(ref_labels_final) & set(pert_labels))
        overall_aris.append(adjusted_rand_score([ref_labels_final[n] for n in common], [pert_labels[n] for n in common]))
        u_ari, c_ari, n_u, n_c = ari_changed_vs_unchanged(ref_labels_final, pert_labels, touched)
        if u_ari is not None: unchanged_aris.append(u_ari)
        if c_ari is not None: changed_aris.append(c_ari)
        n_unchanged_list.append(n_u); n_changed_list.append(n_c)

    print(f"Overall perturbation ARI ({n_seeds_final} seeds): "
          f"{np.mean(overall_aris):.3f} +/- {np.std(overall_aris):.3f}")
    print(f"  Nodes with UNCHANGED incident edges: ARI = "
          f"{np.mean(unchanged_aris):.3f} +/- {np.std(unchanged_aris):.3f} "
          f"(avg n={np.mean(n_unchanged_list):.0f})")
    print(f"  Nodes with CHANGED (removed) incident edges: ARI = "
          f"{np.mean(changed_aris):.3f} +/- {np.std(changed_aris):.3f} "
          f"(avg n={np.mean(n_changed_list):.0f})")
    print(f"  Interpretation: if the method is doing more than reacting to "
          f"noise uniformly, unchanged-node ARI should be visibly higher "
          f"than changed-node ARI.")

    print(f"\n=== Step 3: paired comparison, lambda_sd={lam_sd_grid[0]} vs "
          f"{lam_sd_grid[2]}, same {n_seeds_final} seeds ===")
    paired_a, paired_b = [], []
    for seed in range(n_seeds_final):
        pdata, _ = perturb_data(data, seed=seed, frac=0.10)
        for lam_sd, bucket in [(lam_sd_grid[0], paired_a), (lam_sd_grid[2], paired_b)]:
            ref_l = ref_by_lam[lam_sd][2026]["node_labels_l0"]
            perturbed = run_chain(pdata, alpha, lam_sd, emb_path, order_path)
            pert_l = perturbed[2026]["node_labels_l0"]
            common = sorted(set(ref_l) & set(pert_l))
            bucket.append(adjusted_rand_score([ref_l[n] for n in common], [pert_l[n] for n in common]))
    from scipy.stats import wilcoxon
    try:
        stat, pval = wilcoxon(paired_a, paired_b)
    except ValueError:
        stat, pval = None, None
    print(f"lambda_sd={lam_sd_grid[0]}: mean={np.mean(paired_a):.3f}; "
          f"lambda_sd={lam_sd_grid[2]}: mean={np.mean(paired_b):.3f}; "
          f"Wilcoxon paired test p={pval}")

    print(f"\n=== Step 4: structure-preservation null for real transitions "
          f"(arity-preserving shuffle of NEW edges, {n_seeds_final} seeds) ===")

    def transition_aris(chain):
        out = []
        for prev_c, curr_c in zip(CUTOFFS[:-1], CUTOFFS[1:]):
            l1, l2 = chain[prev_c]["node_labels_l0"], chain[curr_c]["node_labels_l0"]
            common = sorted(set(l1) & set(l2))
            out.append(adjusted_rand_score([l1[n] for n in common], [l2[n] for n in common]))
        return out

    observed_chain = ref_by_lam[best_lam_sd]
    observed_transition_aris = transition_aris(observed_chain)

    growth_null_aris = []  # shape (n_seeds, n_transitions)
    for seed in range(n_seeds_final):
        # Build snapshots by hand: for each transition, take prev's real
        # nodes+edges, and curr's real nodes but with NEW edges shuffled.
        snap_by_cutoff = {}
        for cutoff in CUTOFFS:
            nodes, _ = filter_snapshot(data, cutoff)
            all_edges_cutoff = [e for e in data["hyperedges"]
                                 if e.get("provenance", {}).get("article_year") is not None
                                 and e["provenance"]["article_year"] <= cutoff]
            snap_by_cutoff[cutoff] = {"nodes": nodes, "hyperedges": all_edges_cutoff}

        seed_transition_aris = []
        for prev_c, curr_c in zip(CUTOFFS[:-1], CUTOFFS[1:]):
            shuffled_curr = shuffle_new_edges_arity_preserving(
                snap_by_cutoff[prev_c], snap_by_cutoff[curr_c], seed=seed * 100 + curr_c
            )
            prev_chain_result = run_chain({"nodes": snap_by_cutoff[prev_c]["nodes"],
                                            "hyperedges": snap_by_cutoff[prev_c]["hyperedges"]},
                                           alpha, 0.0, emb_path, order_path, cutoffs=[prev_c])
            shuffled_chain_result = run_chain(shuffled_curr, alpha, 0.0, emb_path, order_path, cutoffs=[curr_c])
            l1 = prev_chain_result[prev_c]["node_labels_l0"]
            l2 = shuffled_chain_result[curr_c]["node_labels_l0"]
            common = sorted(set(l1) & set(l2))
            seed_transition_aris.append(adjusted_rand_score([l1[n] for n in common], [l2[n] for n in common]))
        growth_null_aris.append(seed_transition_aris)
    growth_null_aris = np.array(growth_null_aris)

    print(f"Observed transition ARI (real growth): {[round(a,3) for a in observed_transition_aris]}")
    print(f"Growth-null transition ARI (random new edges), mean per transition: "
          f"{[round(x,3) for x in growth_null_aris.mean(axis=0)]}")
    print(f"Interpretation: if real growth preserves structure more than "
          f"random growth of the same size, observed should exceed the "
          f"growth-null clearly.")

    result = {
        "alpha": alpha,
        "lambda_sd_grid": lam_sd_grid,
        "perturbation_ari_by_lambda_sd_5seed_screen": {str(k): v for k, v in perturbation_results.items()},
        "selected_lambda_sd": best_lam_sd,
        "final_20seed_overall_perturbation_ari": overall_aris,
        "final_20seed_unchanged_node_ari": unchanged_aris,
        "final_20seed_changed_node_ari": changed_aris,
        "paired_test_lambda_a": lam_sd_grid[0], "paired_test_lambda_b": lam_sd_grid[2],
        "paired_test_aris_a": paired_a, "paired_test_aris_b": paired_b,
        "paired_test_wilcoxon_p": pval,
        "observed_transition_aris": [float(a) for a in observed_transition_aris],
        "growth_null_transition_aris_per_seed": growth_null_aris.tolist(),
        "growth_null_transition_ari_mean": growth_null_aris.mean(axis=0).tolist(),
    }
    with open("outputs/lambda_selection_and_null.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved outputs/lambda_selection_and_null.json")
    return best_lam_sd


if __name__ == "__main__":
    main()
