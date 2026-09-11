"""
T5 — Automated faithfulness checker (systematic, not skimmed).

Fix motivating this script: the original pilot judge skimmed the first
several claims per cluster and missed evidence present further down the
list, producing two wrong "overclaim" verdicts. This script instead
searches the FULL claims list for every named entity/term the gloss
mentions, mechanically, so nothing is missed by human/model skimming.

Honest limitation, stated directly: this is a keyword/entity search, not a
real NLI model (no internet access here for DeBERTa-MNLI). It verifies
"this entity is mentioned somewhere in the claims" and reports the
mentioning claim(s) for a human/LLM to do the final relationship check on
a much smaller, pre-filtered set — it does NOT independently verify the
specific relationship/fact asserted about that entity is correct. Treat
its output as high-recall evidence retrieval, not a final verdict.

Three-category classification per named entity in a gloss:
- GROUNDED_AND_SUPPORTED: entity is a cluster member AND appears in at
  least one claim (i.e. there's specific evidence to check by hand).
- GROUNDED_BUT_UNVERIFIABLE: entity is a cluster member but appears in NO
  claim (nothing to check against; not evidence of a problem, just no
  independent signal available).
- NOT_GROUNDED (a real red flag): entity is neither a cluster member nor
  found in any claim — the labeller may have introduced something not
  actually present in its input.

Usage:
    python src/faithfulness_check.py --labelling-input outputs/labelling_input_2026_level0.json --cluster 6 --entities "MACE,SevenNet,GPU parallelization,equivariant"
"""

import argparse
import json


def normalize_text(s: str) -> str:
    """Normalize Unicode hyphen/dash variants to a plain ASCII hyphen
    before matching. Fix (found during rollout): a real member term
    ('Energy‑conserving force field learning', using U+2011 non-breaking
    hyphen) was incorrectly flagged NOT_GROUNDED for the query
    'energy-conserving' (plain ASCII hyphen) — a false alarm from the
    checker, not a real overclaim, caused by a literal Unicode mismatch
    the substring check didn't account for."""
    for dash in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        s = s.replace(dash, "-")
    return s.lower()


def check_entities(cluster_info: dict, entities: list) -> dict:
    all_member_forms = [normalize_text(f) for forms in cluster_info["labeller_input"].values() for f in forms]
    all_claims = cluster_info["faithfulness_signal"]["claim_texts"]

    results = {}
    for entity in entities:
        e_lower = normalize_text(entity)
        is_member = any(e_lower in f or f in e_lower for f in all_member_forms)
        matching_claims = [c for c in all_claims if e_lower in normalize_text(c)]

        if matching_claims:
            category = "GROUNDED_AND_SUPPORTED"
        elif is_member:
            category = "GROUNDED_BUT_UNVERIFIABLE"
        else:
            category = "NOT_GROUNDED"

        results[entity] = {
            "category": category,
            "is_cluster_member": is_member,
            "n_supporting_claims": len(matching_claims),
            "example_claim": matching_claims[0] if matching_claims else None,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelling-input", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--entities", required=True, help="comma-separated list of entities named in the gloss")
    args = parser.parse_args()

    data = json.load(open(args.labelling_input))
    cluster_info = data[args.cluster]
    entities = [e.strip() for e in args.entities.split(",")]

    results = check_entities(cluster_info, entities)
    n_not_grounded = sum(1 for r in results.values() if r["category"] == "NOT_GROUNDED")

    for entity, r in results.items():
        print(f"{entity}: {r['category']} (member={r['is_cluster_member']}, "
              f"n_supporting_claims={r['n_supporting_claims']})")
    if n_not_grounded:
        print(f"\n*** {n_not_grounded} entities NOT_GROUNDED — review for fabrication ***")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
