"""
Assembles outputs/metrics.json from the individual evaluation output files.

Fix (found by external review): metrics.json previously existed but was
built by an ad-hoc inline script, not a checked-in, re-runnable one — a
reproducibility gap. This script is the single source of truth for
metrics.json; re-run it after any evaluation script changes.

Usage:
    python src/assemble_metrics.py
"""

import json
import subprocess


def main():
    coherence = {}
    for cy, path in [
        (2020, "outputs/hierarchy_2020.json"),
        (2022, "outputs/hierarchy_2022.json"),
        (2024, "outputs/hierarchy_2024.json"),
        (2026, "outputs/hierarchy.json"),
    ]:
        out = subprocess.run(["python3", "src/coherence_probe.py", path],
                              capture_output=True, text=True, check=True)
        coherence[cy] = json.loads(out.stdout.strip())

    paper_id = json.load(open("outputs/paper_identity_diagnostic.json"))
    lam = json.load(open("outputs/lambda_selection_and_null.json"))
    t3 = json.load(open("outputs/t3_perturbation_real_embeddings.json"))
    sens = json.load(open("outputs/sensitivity_diagnostics.json"))
    labels = json.load(open("outputs/labelling_faithfulness_pilot.json"))
    extrinsic = json.load(open("outputs/extrinsic_eval_summary.json"))

    # All label blocks: level 0 (4 snapshots) + level 1 (4 snapshots).
    l0_blocks = {2026: ["clusters"], 2020: ["clusters_2020_level0"],
                 2022: ["clusters_2022_level0", "clusters_2022_level0_continued"],
                 2024: ["clusters_2024_level0", "clusters_2024_level0_continued"]}
    label_counts, n_overclaim = {}, 0
    for y, keys in l0_blocks.items():
        entries = [e for k in keys for e in labels[k]]
        label_counts[f"{y}_L0"] = len(entries)
        n_overclaim += sum("overclaim" in str(e.get("faithfulness_verdict", "")).lower()
                           and "no overclaim" not in str(e.get("faithfulness_verdict", "")).lower()
                           for e in entries)
    for y in (2020, 2022, 2024, 2026):
        block = labels[f"level1_{y}"]["clusters"]
        label_counts[f"{y}_L1"] = len(block)
        n_overclaim += sum(str(e.get("verdict", "")).startswith("overclaim") for e in block.values())
    n_labelled = sum(label_counts.values())

    prim = t3["transitions_clustering_only_PRIMARY"]
    sec = t3["transitions_all_nodes_secondary"]
    metrics = {
        "meta": {
            "cutoffs": [2020, 2022, 2024, 2026],
            "alpha": 0.5,
            "linkage": "complete",
            "semantic_model_for_shipped_hierarchies": "all-MiniLM-L6-v2 (real sentence embeddings, via HF Space)",
            "semantic_model_for_t3_null_and_sensitivity": "same MiniLM embeddings (subset matching)",
            "semantic_model_for_extrinsic_scorer": "TF-IDF (label-free scorer; see extrinsic_eval.py)",
            "note": "commit hash not tracked in this environment; record at time of git commit.",
        },
        "coherence": {
            "design": "bibliographic coupling between article pairs via held-out `cites` relation "
                      "(never used in clustering); provenance guard: no article's placement may derive "
                      "from `cites`; permutation null over cluster labels (10,000 permutations)",
            "per_snapshot": coherence,
            "result": "null at every snapshot (best case 2026: z=1.68, p=0.059) -- reported as a null "
                      "result, not significant coherence evidence.",
        },
        "paper_identity_diagnostic": paper_id,
        "stability": {
            "note": "T3's event log (temporal_events.json) is built DIRECTLY from the shipped "
                    "hierarchy_<year>.json files, not a separate re-clustering.",
            "authoritative_perturbation_test": {
                "method": t3["method"],
                "n_seeds_per_snapshot": t3["n_seeds"],
                "primary_node_set": "clustering-placed only (excludes post-hoc attachment)",
                "null_mean_ari_by_snapshot_clustering_only": {
                    c: round(v["ari_mean_clustering_only"], 3) for c, v in t3["null_by_snapshot"].items()},
                "null_ci95_of_mean_by_snapshot_clustering_only": {
                    c: [round(x, 3) for x in v["ari_ci95_of_mean_clustering_only"]]
                    for c, v in t3["null_by_snapshot"].items()},
                "k0_stayed_within_p2_target_all_seeds_all_snapshots": all(
                    v["perturbed_k0_all_within_p2"] for v in t3["null_by_snapshot"].values()),
                "transitions_clustering_only_PRIMARY": prim,
                "transitions_all_nodes_secondary": sec,
                "n_transitions_significant_primary_holm_0.05": t3["n_transitions_significant_primary_holm_0.05"],
                "interpretation": t3["interpretation"] + " " + t3["caveat_not_equivalent_sized"],
            },
            "superseded": {
                "2026_only_null": "Earlier versions tested every transition against the 2026 null only; "
                                  "2022->2024 then gave p=0.03 (primary), which (a) did not survive "
                                  "multiple-testing correction (Holm/BH adjusted p=0.089) and (b) used a "
                                  "null ~0.26 ARI less stable than the 2022/2024 snapshots' own nulls. "
                                  "Kept in t3_perturbation_real_embeddings.json as p_vs_2026_null_SUPERSEDED.",
                "lambda_sweep_file": "outputs/lambda_selection_and_null.json is LEGACY: produced by "
                                     "src/temporal_reg.py (raw-edge perturbation, which breaks embedding "
                                     "subset matching; 13/20 seeds collapse to ARI=0) on an older pipeline "
                                     "state; its observed transition ARIs (0.82/0.44/0.50) do not match the "
                                     "shipped hierarchies (see above), so its growth-null comparison is NOT "
                                     "reported as evidence. Only its decision (lambda_sd=0, i.e. no temporal "
                                     "regularisation) is used, and that decision is the untuned default.",
                "selected_lambda_sd": lam.get("selected_lambda_sd"),
            },
        },
        "method_sensitivity": {
            "source": "src/sensitivity_diagnostics.py",
            "rank_normalisation_ties": {
                c: {k: v[k] for k in ("structural_zero_fraction", "tied_zero_rank", "smallest_nonzero_rank",
                                      "structural_variance_share_shipped", "structural_variance_share_zero_pinned",
                                      "ari_shipped_vs_zero_pinned_levels_0_2",
                                      "own_10pct_removal_noise_band_L0_2.5_97.5pct")}
                for c, v in sens["per_snapshot"].items()},
            "assignment_tie_breaking": {
                c: {k: v[k] for k in ("tie_share_of_primary", "flipped_under_permutation_mean",
                                      "flipped_union_subset_of_tied", "node_laminar_all_permutations")}
                for c, v in sens["per_snapshot"].items()},
            "canonical_tiebreak_changes_labelled_n_members": sens["canonical_tiebreak_changes_labelled_n_members"],
            "decision": "documented, not changed: both fixes re-cluster/re-assign every snapshot and invalidate "
                        "existing labels; see report.md, Limitations.",
        },
        "labels": {
            "n_labelled_clusters": n_labelled,
            "by_snapshot_level": label_counts,
            "n_overclaims_uncorrected": n_overclaim,
            "over_claim_rate": n_overclaim / n_labelled,
            "method": "LLM labeller sees only member surface forms; each named entity in a gloss is checked "
                      "against the member list and against independent `claims` text (article_year <= cutoff) "
                      "with src/faithfulness_check.py. Entity-level check, not NLI of relationships.",
            "level1_2026_entity_summary": labels["level1_2026"]["entity_summary"],
            "levels_2_3": "deterministic template labels from verbatim member forms (label_source='template'); "
                          "outside the T5 faithfulness scope",
            "legacy_pilot_2026_L0_detail": labels["clusters"],
        },
        "extrinsic": extrinsic,
        "verified_vs_assumed": {
            "verified_directly_by_rerunning": [
                "All four hierarchies regenerate deterministically from method.py (edge labels unchanged by "
                "the authored_by attachment fix; only 105/131/250/318 previously-unassigned author nodes change)",
                "Level-0 counts 11/15/15/14 (P2) and node-level laminarity PASS on all 4 snapshots; laminarity "
                "also holds under 20 random edge-order permutations per snapshot",
                "Embedding scripts' edge filter equals method.filter_snapshot at every cutoff "
                "(tests/test_filter_consistency.py), and the shipped 334-row embedding order equals the 2026 edge order",
                "Every label's n_members matches the current hierarchy (guard in merge_supernodes.py)",
                "Persistent ids recomputed by merge_supernodes.py reproduce temporal_events.json exactly",
                "Coherence probe z/p unchanged by the attachment fix (authors never touch `cites`)",
                "T3 snapshot-matched nulls: each reference clustering equals the shipped hierarchy (asserted)",
            ],
            "assumed_or_not_yet_verified": [
                "Label faithfulness is checked at entity level; asserted relationships are not NLI-verified, and "
                "labeller and checker were the same model family (procedural, not model-level, separation)",
                "Extrinsic task is a NULL/uninformative result with a TF-IDF scorer (0/47 hierarchical, 2/47 flat)",
                "Coherence probe has low power (9-37 articles per snapshot)",
                "10%-edge-removal is not a matched-magnitude null for 40-92% real growth",
                "persistent_id is tracked at level 0 only",
            ],
        },
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Saved outputs/metrics.json")


if __name__ == "__main__":
    main()
