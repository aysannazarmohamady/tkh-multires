"""
T5 — Labelling with measured faithfulness: input preparation.

Produces, per cluster (a given snapshot + level):
1. `labeller_input`: member surface forms grouped by node type. This is
   what the labeller (an LLM) sees when generating a label + gloss. It is
   exactly the content that also drove clustering (member surface forms),
   which is unavoidable for a *topical* label — but see (2).
2. `faithfulness_signal`: `claims` text (independent, per §6's circularity
   warning) associated with the cluster's articles, filtered to
   `article_year <= cutoff` (P6 temporal honesty) — this is NEVER shown to
   the labeller, only used afterward to check the generated gloss.

The two are kept in strictly separate fields so a labelling script cannot
accidentally leak (2) into (1).

Usage:
    python src/prepare_labelling_input.py --hierarchy outputs/hierarchy.json --cutoff 2026 --level 0
"""

import argparse
import json
from collections import defaultdict


def build_cluster_inputs(hierarchy_path: str, data_path: str, cutoff: int, level: int) -> dict:
    data = json.load(open(data_path))
    hier = json.load(open(hierarchy_path))
    node_lookup = {n["id"]: n for n in data["nodes"]}

    node_to_cluster = {nid: v[level] for nid, v in hier["node_assignment"].items() if v[level] is not None}

    # cluster -> member node ids (excluding claim/cited_work/author, which
    # are attached post-hoc and are metadata, not topical content)
    cluster_members = defaultdict(list)
    for nid, cid in node_to_cluster.items():
        ntype = node_lookup.get(nid, {}).get("type")
        if ntype in ("claim", "cited_work", "author"):
            continue
        cluster_members[cid].append(nid)

    # Fix (found while inspecting real output): associate a cluster with
    # articles via the EDGE-level provenance.article_id of the actual
    # clustering hyperedges assigned to it, NOT via a member node's
    # provenance.articles list. The latter is too broad — a shared dataset
    # or technique node's `provenance.articles` lists every paper that ever
    # mentions it, which pulled in claims from unrelated papers (e.g. a
    # cluster about one paper's tensorial Atomic Cluster Expansion method
    # pulled in claims about an unrelated paper, "MACE", because both
    # papers' provenance touched a shared "acetylacetone" dataset node).
    # Edge-level article_id is precise: it's the one paper that actually
    # produced that specific edge.
    edge_ids = hier.get("edge_ids")
    edge_labels_all = hier.get("edge_cluster_labels")
    cluster_articles = defaultdict(set)
    if edge_ids and edge_labels_all:
        edge_labels = edge_labels_all[level]
        edge_lookup = {e["id"]: e for e in data["hyperedges"]}
        for eid, cid in zip(edge_ids, edge_labels):
            e = edge_lookup.get(eid)
            if e is None:
                continue  # merged evaluated_on edges aren't in raw hyperedges; skip
            aid = e.get("provenance", {}).get("article_id")
            if aid is not None:
                cluster_articles[cid].add(aid)

    # article_id -> claim texts (article_year <= cutoff only, per P6)
    article_claims = defaultdict(list)
    for e in data["hyperedges"]:
        if e["relation_type"] != "claims":
            continue
        article_year = e.get("provenance", {}).get("article_year")
        if article_year is None or article_year > cutoff:
            continue
        article_id = e.get("provenance", {}).get("article_id")
        claim_ids = [m for m in e["members"] if node_lookup.get(m, {}).get("type") == "claim"]
        for cid_node in claim_ids:
            article_claims[article_id].append(node_lookup[cid_node]["surface_form"])

    result = {}
    for cid, member_ids in cluster_members.items():
        by_type = defaultdict(list)
        for nid in member_ids:
            n = node_lookup.get(nid)
            if n:
                by_type[n["type"]].append(n["surface_form"])

        claim_texts = []
        for aid in cluster_articles.get(cid, []):
            claim_texts.extend(article_claims.get(aid, []))

        result[str(cid)] = {
            "n_members": len(member_ids),
            "labeller_input": {t: sorted(set(forms)) for t, forms in by_type.items()},
            "faithfulness_signal": {
                "n_articles": len(cluster_articles.get(cid, [])),
                "claim_texts": claim_texts,
            },
        }
    return result


def check_grounding(gloss: str, labeller_input: dict) -> dict:
    """Deterministic grounding check: every named entity the gloss
    mentions should verbatim-match (case-insensitive substring) a member
    surface form. Returns which surface forms were 'used' and a rough flag
    for unmatched capitalized/technical-looking tokens the gloss didn't
    ground. This is a coarse, mechanical check — not a substitute for the
    claims-based faithfulness check, but a cheap first pass per the
    verbatim-entity design principle (avoid asserting datasets/methods/
    comparisons not literally in the member list)."""
    all_forms = [f.lower() for forms in labeller_input.values() for f in forms]
    gloss_lower = gloss.lower()
    grounded_forms = [f for f in all_forms if f in gloss_lower]
    return {"grounded_member_forms_referenced": sorted(set(grounded_forms))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", default="outputs/hierarchy.json")
    parser.add_argument("--data", default="data/tkh_collection10.json")
    parser.add_argument("--cutoff", type=int, default=2026)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", default="outputs/labelling_input.json")
    args = parser.parse_args()

    result = build_cluster_inputs(args.hierarchy, args.data, args.cutoff, args.level)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Built labelling input for {len(result)} clusters (cutoff={args.cutoff}, level={args.level})")
    for cid, info in sorted(result.items(), key=lambda kv: -kv[1]["n_members"])[:3]:
        print(f"  cluster {cid}: {info['n_members']} members, "
              f"{info['faithfulness_signal']['n_articles']} articles, "
              f"{len(info['faithfulness_signal']['claim_texts'])} claim texts available for faithfulness check")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
