"""
Merge labels/glosses and persistent ids into the shipped hierarchy files,
producing the per-super-node schema required by assessment §6.2:

    {id, level, parent_id, member_ids, label, gloss, persistent_id}

plus a few non-required but useful fields (snapshot, cluster_label,
child_ids, edge_ids, member_provenance, label_source, faithfulness_verdict).

Sources (all read-only; nothing is re-clustered):
  - outputs/hierarchy_<year>.json / outputs/hierarchy.json (2026)
  - outputs/labelling_faithfulness_pilot.json  (labels for L0 + L1)
  - src/build_temporal_events.py               (persistent ids, L0 only)

Guards (fail loudly rather than ship a silently inconsistent artifact):
  - every labelled cluster id must exist in its hierarchy level;
  - each label's n_members must equal the current hierarchy's count
    (same definition as prepare_labelling_input.py) -> catches labels
    made on a stale hierarchy after any re-run of method.py;
  - persistent ids are recomputed with build_temporal_events.classify_events
    and the resulting events must equal the shipped temporal_events.json
    (ignoring the t4_cohesion enrichment) -> catches drift;
  - parent_id derived from edge_cluster_labels must agree with every
    member node's assignment path (node-level laminarity);
  - child member sets must be subsets of parent member sets.

Usage:
    python src/merge_supernodes.py                 # writes in place
    python src/merge_supernodes.py --out-dir /tmp/check
    python src/merge_supernodes.py --template-unlabelled --strict
        # deterministic, member-only template labels for every unlabelled
        # super-node at level >= 2 (label_source="template"); --strict
        # fails unless EVERY super-node ends up with a label and a gloss.
    (--template-singletons is kept as an alias that templates level 3 only.)

Scope decision (task §6.2 vs T5): §6.2 asks for a label and gloss on every
super-node; T5 asks for generated labels with measured faithfulness at
levels 0 and 1. Levels 0-1 therefore carry LLM labels checked by
faithfulness_check.py (label_source="llm_pilot"); levels 2-3 carry template
labels built only from verbatim member surface forms and relation types
(label_source="template"), which cannot over-claim by construction but are
not "semantic" summaries. persistent_id is tracked at level 0 only (the
temporal_events.json matching level); it is null at levels >= 1.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
import build_temporal_events as bte  # noqa: E402

SCHEMA_VERSION = "6.2-v1"
HIERARCHY_PATHS = dict(bte.HIERARCHY_PATHS)  # {year: path}; single source of truth
LABELS_PATH = "outputs/labelling_faithfulness_pilot.json"
EVENTS_PATH = "outputs/temporal_events.json"
DATA_PATH = "data/tkh_collection10.json"

# The pilot file stores clusters under heterogeneous keys; map them explicitly
# so an unexpected key layout fails instead of being silently skipped.
LABEL_SOURCES = {
    (2026, 0): ["clusters"],
    (2020, 0): ["clusters_2020_level0"],
    (2022, 0): ["clusters_2022_level0", "clusters_2022_level0_continued"],
    (2024, 0): ["clusters_2024_level0", "clusters_2024_level0_continued"],
    (2020, 1): ["level1_2020.clusters"],
    (2022, 1): ["level1_2022.clusters"],
    (2024, 1): ["level1_2024.clusters"],
    (2026, 1): ["level1_2026.clusters"],  # absent in current upload -> reported, not fatal
}
NON_TOPICAL_TYPES = ("claim", "cited_work", "author")  # as in prepare_labelling_input.py


def sn_id(year: int, level: int, cluster_label: int) -> str:
    return f"{year}/L{level}/c{cluster_label}"


def _get_path(obj: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def load_labels(path: str) -> tuple[dict, list]:
    """Returns ({(year, level): {cluster_label: entry}}, [missing source keys])."""
    raw = json.load(open(path))
    out, missing = {}, []
    for key, dotted_keys in LABEL_SOURCES.items():
        merged = {}
        for dk in dotted_keys:
            block = _get_path(raw, dk)
            if block is None:
                missing.append(dk)
                continue
            items = ([dict(v, cluster_id=int(k)) for k, v in block.items()]
                     if isinstance(block, dict) else block)
            for e in items:
                cid = int(e["cluster_id"])
                if cid in merged:
                    raise ValueError(f"Duplicate label for {key} cluster {cid} (source {dk})")
                merged[cid] = e
        if merged:
            out[key] = merged
    return out, missing


def recompute_persistent_ids(shipped_events_path: str) -> dict:
    """{year: {l0_cluster_label: persistent_id}}, recomputed exactly as in
    build_temporal_events.main(), then checked against the shipped events."""
    cutoffs = sorted(HIERARCHY_PATHS)
    chain = bte.load_chain_from_shipped_hierarchies(HIERARCHY_PATHS)
    counter = [0]
    first = cutoffs[0]
    prev_ids = {}
    for cc in bte.cluster_edge_sets(chain[first]["edge_labels_l0"]):
        prev_ids[cc] = f"p{counter[0]}"
        counter[0] += 1
    id_maps = {first: dict(prev_ids)}
    events = {first: [{"type": "birth", "persistent_id": p, "curr_cluster_label": c}
                      for c, p in prev_ids.items()]}
    for a, b in zip(cutoffs[:-1], cutoffs[1:]):
        curr_ids, ev = bte.classify_events(chain[a]["edge_labels_l0"], chain[b]["edge_labels_l0"],
                                           prev_ids, counter)
        id_maps[b], events[b], prev_ids = curr_ids, ev, curr_ids

    shipped = json.load(open(shipped_events_path))["events_by_snapshot"]

    def strip(evs):
        # enrich_events_with_t4_cohesion adds t4_cohesion / t4_cohesion_of_dest_clusters
        return [{k: v for k, v in e.items() if not k.startswith("t4_cohesion")} for e in evs]

    def canon(evs):  # JSON round-trip: int dict keys -> str, as in the shipped file
        return json.loads(json.dumps(strip(evs)))

    for y in cutoffs:
        if canon(events[y]) != strip(shipped[str(y)]):
            raise RuntimeError(f"Recomputed temporal events for {y} differ from {shipped_events_path}; "
                               f"re-run src/build_temporal_events.py before merging.")
    for y, m in id_maps.items():
        dup = [p for p, n in Counter(m.values()).items() if n > 1]
        if dup:
            raise RuntimeError(f"{y}: persistent id(s) {dup} assigned to >1 level-0 cluster")
    return id_maps


def template_singleton_label(edge: dict, node_lookup: dict, max_terms: int = 4) -> tuple[str, str]:
    forms = [node_lookup[m]["surface_form"] for m in edge["members"] if m in node_lookup]
    head = ", ".join(forms[:max_terms]) + (" …" if len(forms) > max_terms else "")
    return (f"{edge['relation_type']}: {head}",
            f"Single hyperedge ({edge['relation_type']}, {len(forms)} members): {', '.join(forms)}.")


def template_cluster_label(edges: list, node_lookup: dict, max_terms: int = 4) -> tuple[str, str]:
    """Deterministic, member-only label for a multi-hyperedge super-node:
    most frequent member surface forms (ties -> alphabetical) + relation mix."""
    if len(edges) == 1:
        return template_singleton_label(edges[0], node_lookup, max_terms)
    freq = Counter(node_lookup[m]["surface_form"] for e in edges for m in e["members"] if m in node_lookup)
    top = [f for f, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]]
    rels = Counter(e["relation_type"] for e in edges)
    rel_txt = ", ".join(f"{r} x{n}" for r, n in sorted(rels.items(), key=lambda kv: (-kv[1], kv[0])))
    return (" / ".join(top),
            f"Template: {len(edges)} hyperedges ({rel_txt}) whose most frequent members are {', '.join(top)}.")


def build_super_nodes(year: int, hier: dict, labels: dict, pid_map: dict, node_lookup: dict,
                      edge_lookup: dict, template_min_level: int | None) -> list:
    edge_ids = hier["edge_ids"]
    levels = hier["edge_cluster_labels"]
    n_levels = len(levels)
    assign = hier["node_assignment"]
    prov = hier.get("node_assignment_provenance", {})

    # Edge-derived parent map (the dendrogram is laminar at edge level).
    parent_of = [dict() for _ in range(n_levels)]
    for lvl in range(1, n_levels):
        for coarse, fine in zip(levels[lvl - 1], levels[lvl]):
            prev = parent_of[lvl].setdefault(fine, coarse)
            if prev != coarse:
                raise RuntimeError(f"{year}: edge-level laminarity broken at L{lvl} cluster {fine}")

    edges_in = [defaultdict(list) for _ in range(n_levels)]
    for i, eid in enumerate(edge_ids):
        for lvl in range(n_levels):
            edges_in[lvl][levels[lvl][i]].append(eid)

    members_in = [defaultdict(list) for _ in range(n_levels)]
    for nid in sorted(assign):
        path = assign[nid]
        if path[0] is None:
            continue
        for lvl in range(n_levels):
            members_in[lvl][path[lvl]].append(nid)
            if lvl > 0 and parent_of[lvl].get(path[lvl]) != path[lvl - 1]:
                raise RuntimeError(f"{year}: node {nid} path disagrees with edge-derived parent at L{lvl}")

    children = [defaultdict(list) for _ in range(n_levels)]
    for lvl in range(1, n_levels):
        for fine, coarse in parent_of[lvl].items():
            children[lvl - 1][coarse].append(fine)

    out = []
    for lvl in range(n_levels):
        lab = labels.get((year, lvl), {})
        for cl in sorted(set(levels[lvl])):
            members = members_in[lvl].get(cl, [])
            entry = lab.get(cl)
            label = gloss = verdict = source = None
            if entry is not None:
                n_topical = sum(1 for m in members if node_lookup[m].get("type") not in NON_TOPICAL_TYPES)
                if entry.get("n_members") is not None and entry["n_members"] != n_topical:
                    raise RuntimeError(f"Stale label: {year} L{lvl} c{cl} labelled with n_members="
                                       f"{entry['n_members']} but hierarchy has {n_topical}")
                label, gloss, source = entry["label"], entry["gloss"], "llm_pilot"
                verdict = entry.get("faithfulness_verdict", entry.get("verdict"))
            elif template_min_level is not None and lvl >= template_min_level:
                label, gloss = template_cluster_label([edge_lookup[e] for e in edges_in[lvl][cl]], node_lookup)
                source = "template"
            out.append({
                "id": sn_id(year, lvl, cl),
                "snapshot": year,
                "level": lvl,
                "cluster_label": int(cl),
                "parent_id": sn_id(year, lvl - 1, parent_of[lvl][cl]) if lvl > 0 else None,
                "child_ids": [sn_id(year, lvl + 1, c) for c in sorted(children[lvl].get(cl, []))],
                "member_ids": members,
                "member_provenance": dict(Counter(prov.get(m, "unknown") for m in members)),
                "edge_ids": edges_in[lvl][cl],
                "label": label,
                "gloss": gloss,
                "label_source": source,
                "faithfulness_verdict": verdict,
                "persistent_id": pid_map.get(cl) if lvl == 0 else None,
            })

    # Laminar containment on the delivered members.
    by_id = {s["id"]: s for s in out}
    for s in out:
        if s["parent_id"] and not set(s["member_ids"]) <= set(by_id[s["parent_id"]]["member_ids"]):
            raise RuntimeError(f"{s['id']}: members not contained in parent {s['parent_id']}")
        if s["level"] == 0 and s["persistent_id"] is None:
            raise RuntimeError(f"{s['id']}: level-0 super-node has no persistent_id")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="default: overwrite hierarchy files in place")
    ap.add_argument("--template-singletons", action="store_true", help="template labels for level 3 only")
    ap.add_argument("--template-unlabelled", action="store_true",
                    help="template labels for every unlabelled super-node at level >= 2")
    ap.add_argument("--strict", action="store_true",
                    help="fail unless every super-node has a label and a gloss")
    args = ap.parse_args()
    template_min_level = 2 if args.template_unlabelled else (3 if args.template_singletons else None)

    data = json.load(open(DATA_PATH))
    node_lookup = {n["id"]: n for n in data["nodes"]}
    edge_lookup = {e["id"]: e for e in data["hyperedges"]}
    # merged evaluated_on edges exist only inside method.py's pipeline
    from method import filter_snapshot
    for y in HIERARCHY_PATHS:
        for e in filter_snapshot(data, y)[1]:
            edge_lookup.setdefault(e["id"], e)

    labels, missing_sources = load_labels(LABELS_PATH)
    pid_maps = recompute_persistent_ids(EVENTS_PATH)

    for year, path in sorted(HIERARCHY_PATHS.items()):
        hier = json.load(open(path))
        known = {(year, l): set(ls) for l, ls in enumerate(hier["edge_cluster_labels"])}
        for (y, l), entries in labels.items():
            if y == year:
                extra = set(entries) - known[(y, l)]
                if extra:
                    raise RuntimeError(f"Labels for {y} L{l} reference unknown clusters {sorted(extra)}")
        hier.pop("super_nodes", None)  # idempotent: never build on a previous merge's output
        sns = build_super_nodes(year, hier, labels, pid_maps[year], node_lookup, edge_lookup,
                                template_min_level)
        missing = [s["id"] for s in sns if not s["label"] or not s["gloss"]]
        if args.strict and missing:
            raise RuntimeError(f"{year}: {len(missing)} super-nodes lack label/gloss (e.g. {missing[:5]})")
        hier["super_node_schema_version"] = SCHEMA_VERSION
        hier["super_nodes"] = sns
        dest = Path(args.out_dir, Path(path).name) if args.out_dir else Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(hier, f, indent=2, sort_keys=True)
        cov = Counter((s["level"], s["label"] is not None) for s in sns)
        per_level = ", ".join(f"L{l}: {cov[(l, True)]}/{cov[(l, True)] + cov[(l, False)]}"
                              for l in range(len(hier["edge_cluster_labels"])))
        empty = sum(1 for s in sns if not s["member_ids"])
        print(f"{year}: {len(sns)} super-nodes -> {dest}  labelled {per_level}; empty-member super-nodes: {empty}")
    if missing_sources:
        print(f"WARNING: label source keys absent from {LABELS_PATH}: {missing_sources}")


if __name__ == "__main__":
    main()
