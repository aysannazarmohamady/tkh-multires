"""
T5 — Check a label draft with src/faithfulness_check.py and merge it into
outputs/labelling_faithfulness_pilot.json as a `level<k>_<year>` block.

A draft (labels/draft_<year>_level<k>.json) maps cluster id -> {label,
gloss, entities}, where `entities` lists EVERY named entity/term the gloss
asserts. Each entity is classified by check_entities against the cluster's
member list and its independent claims signal. Verdict rules:
  - any NOT_GROUNDED entity            -> "overclaim"   (counts toward over-claim rate)
  - all entities grounded              -> "faithful (entity-level)"
  - label starts a catch-all marker    -> additionally flagged "broad"
The check is entity-level (high-recall evidence retrieval), NOT an NLI check
of the asserted relationships; see faithfulness_check.py's docstring.

Refuses to merge if the draft does not cover exactly the clusters in the
labelling input, or if the target block already exists (use --overwrite).

Usage:
    python src/merge_label_draft.py --draft labels/draft_2026_level1.json \
        --labelling-input outputs/labelling_input_2026_level1.json --key level1_2026
"""

import argparse
import json
import sys

sys.path.insert(0, "src")
from faithfulness_check import check_entities  # noqa: E402

PILOT = "outputs/labelling_faithfulness_pilot.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--labelling-input", required=True)
    ap.add_argument("--key", required=True, help="e.g. level1_2026")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    draft = json.load(open(args.draft))
    meta = draft.pop("_meta", {})
    li = json.load(open(args.labelling_input))
    if set(draft) != set(li):
        raise SystemExit(f"draft/input cluster mismatch: missing={sorted(set(li) - set(draft), key=int)}, "
                         f"extra={sorted(set(draft) - set(li), key=int)}")
    pilot = json.load(open(PILOT))
    if args.key in pilot and not args.overwrite:
        raise SystemExit(f"{args.key} already present in {PILOT}; pass --overwrite to replace it")

    clusters, n_over, n_broad, n_ent, n_sup = {}, 0, 0, 0, 0
    for cid in sorted(draft, key=int):
        d, info = draft[cid], li[cid]
        res = check_entities(info, d["entities"])
        bad = sorted(e for e, r in res.items() if r["category"] == "NOT_GROUNDED")
        sup = sum(r["category"] == "GROUNDED_AND_SUPPORTED" for r in res.values())
        broad = "catch-all" in d["gloss"].lower() or "(broad)" in d["label"].lower()
        verdict = "overclaim" if bad else "faithful (entity-level)"
        if broad:
            verdict += "; broad catch-all cluster"
        n_over += bool(bad)
        n_broad += broad
        n_ent += len(res)
        n_sup += sup
        clusters[cid] = {
            "n_members": info["n_members"],
            "n_articles": info["faithfulness_signal"]["n_articles"],
            "label": d["label"], "gloss": d["gloss"], "verdict": verdict,
            "entity_check": {"n_entities": len(res), "n_supported_by_claims": sup,
                             "n_member_only": len(res) - sup - len(bad), "not_grounded": bad},
        }

    n = len(clusters)
    pilot[args.key] = {
        "coverage": f"{n}/{n} (COMPLETE)",
        "labeller": meta.get("labeller"),
        "labelling_rule": meta.get("rule"),
        "clusters": clusters,
        "over_claim_rate": n_over / n,
        "entity_summary": {"n_entities_checked": n_ent, "n_supported_by_claims": n_sup,
                           "n_broad_catch_all_clusters": n_broad},
        "method_note": "entity-level check via src/faithfulness_check.py (member list + independent claims); "
                       "relationships asserted between entities were not independently NLI-verified",
    }
    with open(PILOT, "w") as f:
        json.dump(pilot, f, indent=2, ensure_ascii=False)
    print(f"{args.key}: {n} clusters merged; over-claim rate {n_over}/{n}; {n_broad} broad; "
          f"{n_sup}/{n_ent} named entities also found in claims")


if __name__ == "__main__":
    main()
