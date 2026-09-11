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
    t3_real = json.load(open("outputs/t3_perturbation_real_embeddings.json"))
    labels = json.load(open("outputs/labelling_faithfulness_pilot.json"))
    extrinsic = json.load(open("outputs/extrinsic_eval_summary.json"))

    metrics = {
        "meta": {
            "cutoffs": [2020, 2022, 2024, 2026],
            "alpha": 0.5,
            "linkage": "complete",
            "semantic_model_for_shipped_hierarchies": "all-MiniLM-L6-v2 (real sentence embeddings, via HF Space)",
            "semantic_model_for_perturbation_and_extrinsic_scripts": "TF-IDF (offline; see script docstrings for why real embeddings aren't used there)",
            "note": "commit hash not tracked in this environment; record at time of git commit. "
                    "requirements.txt confirmed to install cleanly in an independent clean virtualenv "
                    "by an external reviewer.",
        },
        "coherence": {
            "design": "bibliographic coupling between article pairs via held-out `cites` relation "
                      "(never used in clustering); provenance guard: no article's placement may derive "
                      "from `cites` (relaxed from `provenance == 'primary'` to `!= 'cites'` since articles "
                      "may also be placed deterministically via `presents`)",
            "per_snapshot": coherence,
            "result": "null at every snapshot (best case 2026: z=1.68, p=0.059) — reported as a null "
                      "result, not significant coherence evidence.",
        },
        "paper_identity_diagnostic": paper_id,
        "stability": {
            "note": "T3's event log (temporal_events.json) is built DIRECTLY from the shipped "
                    "hierarchy_<year>.json files, not a separate re-clustering.",
            "authoritative_perturbation_test": {
                "method": "src/t3_perturbation_real_embeddings.py — perturbs post-merge "
                          "clustering-eligible edges (10% removed, 100 seeds), real MiniLM "
                          "embeddings via subset matching; corrected statistical test (each "
                          "transition vs. the distribution of individual seed ARIs, not the "
                          "CI of the mean)",
                "primary_node_set": "clustering-placed only (excludes post-hoc attachment)",
                "transition_aris_clustering_only": t3_real["transition_aris_clustering_only"],
                "transition_p_values_clustering_only": t3_real["transition_p_values_clustering_only_PRIMARY"],
                "transition_aris_all_nodes_secondary": t3_real["transition_aris_all_nodes"],
                "transition_p_values_all_nodes_secondary": t3_real["transition_p_values_all_nodes"],
                "k0_stayed_within_p2_target_all_seeds": t3_real["perturbed_k0_all_within_p2"],
                "interpretation": "Only the 2022->2024 transition is robustly more stable than "
                                  "10%-edge-removal noise (p=0.03 primary, p=0.01 secondary); "
                                  "2020->2022 and 2024->2026 are not distinguishable from noise "
                                  "by this test. A partial, not uniform, P5 result. Not an "
                                  "equivalent-magnitude comparison (perturbation removes 10% of "
                                  "edges; real growth adds 40-92%).",
            },
            "lambda_decision": {
                "note": "lambda=0 was chosen after an earlier (TF-IDF-based, degenerate) "
                        "perturbation analysis; that analysis is NOT the authoritative "
                        "perturbation result above, and should be read as 'lambda=0 is an "
                        "untuned default that happened not to be contradicted', not as a "
                        "result that positively informed the choice.",
                "selected_lambda_sd": lam.get("selected_lambda_sd"),
                "raw_lambda_sweep_file": "outputs/lambda_selection_and_null.json (TF-IDF-based, "
                                         "degenerate — see file for caveats)",
            },
        },
        "labels": {
            "pilot_n_labelled": len(labels["clusters"]),
            "pilot_over_claim_rate": labels["over_claim_rate"],
            "pilot_note": labels["over_claim_rate_note"],
            "pilot_detail": labels["clusters"],
            "full_scale_run": "NOT YET DONE — pilot only (4/14 level-0 clusters, 2026 snapshot).",
        },
        "extrinsic": extrinsic,
        "verified_vs_assumed": {
            "verified_directly_by_rerunning": [
                "T1 snapshot counts and temporal-leak fix",
                "T2 level-0 cluster counts (10-15 target) on all 4 snapshots",
                "Node-level laminarity PASS on all 4 snapshots",
                "Coherence probe z/p values, reproduced exactly by an independent external review",
                "T3/T4: events and cohesion both derive from the same shipped hierarchies, "
                "reproduced byte-identically by an independent external review",
                "T5 pilot: corrected to 0/4 overclaims after external review found both original "
                "verdicts were wrong (evidence was present but missed on an incomplete read)",
                "T6: cost accounting AND ranking determinism corrected -- an independent review "
                "traced a specific 'hit' to a hash-randomization artifact among zero-similarity "
                "ties (verified: the hit node, cite_00001, has TF-IDF score exactly 0.0)",
                "T3 authoritative perturbation test: real MiniLM embeddings via subset matching "
                "after the evaluated_on merge, 100 seeds, all k0 within P2 target, ARI closely "
                "matched an independent external re-run once using the same node set",
                "eff_first_seen fix: 17 nodes (e.g. NequIP) where an edge's article_year predated "
                "the node's own first_seen_year now correctly included from the earlier snapshot; "
                "dropped_partial_edges is now 0 at every cutoff (was previously nonzero at 2020/2022/2024)",
            ],
            "assumed_or_not_yet_verified": [
                "Full-scale T5 labelling (only a 4-cluster pilot exists, corrected to 0/4 "
                "overclaims, still too small a sample for a real rate; task requires 'more than "
                "eyeballing ten')",
                "Extrinsic task result is now a confirmed NULL/uninformative result with TF-IDF "
                "(0/47 hierarchical, 2/47 flat found) -- a denser embedding scorer is needed "
                "before this comparison says anything about the hierarchy's value",
                "hierarchy_*.json files do not yet contain the full per-super-node record schema "
                "(id, level, parent_id, member_ids, label, gloss, persistent_id) that section 6.2 "
                "of the task specifies",
            ],
        },
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved outputs/metrics.json")


if __name__ == "__main__":
    main()
