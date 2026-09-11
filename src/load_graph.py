"""
T1 — Load and describe the evolving TKH graph.

Cumulative (not disjoint) snapshots are used, since the task requires a
sequence of graph states over time, not independent yearly slices — later
snapshots must contain earlier ones for growth/stability comparisons in T3
to make sense.

Snapshot membership field, corrected (temporal-leak fix, found by external
review): nodes are filtered on `first_seen_year` (when the entity actually
entered THIS corpus), not `year`/`origin_year` (when the entity was
invented/published in the real world, which can predate its appearance
here). An earlier version filtered on `year`, which falls back to
`origin_year` when known — this let entities into a snapshot before the
corpus itself had any record of them. Concretely: 334 of 1,839 nodes in the
old H(2020) had `first_seen_year > 2020`, meaning the corpus did not
actually contain any trace of them until later, even though their
real-world origin predated 2020. This is exactly the kind of "knowledge
from the future" leak P6 (temporal honesty) warns against, just at the
snapshot-construction level rather than the labeling level. Edges are
filtered on `provenance.article_year` (the year of the article that
produced the edge) for the same reason, not the edge's own `year` field,
which can differ from `article_year` for a similar reason.

An edge is only included once ALL of its member nodes are present in the
snapshot (by `first_seen_year`). This is what surfaces
`dropped_partial_edges`, a genuine data-quality signal, not merely an
artifact of the inclusion rule.

Usage:
    python src/load_graph.py --data data/tkh_collection10.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_CUTOFFS = [2020, 2022, 2024, 2026]


def load_tkh(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_eff_first_seen(data: dict) -> dict:
    """{node_id: effective_first_seen_year} = min(the node's own
    `first_seen_year`, the earliest `provenance.article_year` among edges
    that mention it).

    Fix (found by external review): 17 nodes (e.g. NequIP: recorded
    `first_seen_year=2023`, but mentioned by an edge from a 2021 article)
    have a `first_seen_year` LATER than an edge that actually references
    them — an export inconsistency, not a real "this entity only entered
    the corpus later" fact. Using the raw `first_seen_year` for these 17
    nodes was itself a small residual temporal-leak/inconsistency: it
    could exclude a node from a snapshot that already, correctly, contains
    an edge mentioning it (which would then be among the
    `dropped_partial_edges`). Using the earliest of the two sources fixes
    this for exactly the affected nodes and is a no-op for the other
    5,781.
    """
    earliest_edge_year = {}
    for e in data["hyperedges"]:
        ay = e.get("provenance", {}).get("article_year")
        if ay is None:
            continue
        for m in e["members"]:
            if m not in earliest_edge_year or ay < earliest_edge_year[m]:
                earliest_edge_year[m] = ay

    eff = {}
    for n in data["nodes"]:
        fsy = n.get("first_seen_year")
        eey = earliest_edge_year.get(n["id"])
        if fsy is None:
            eff[n["id"]] = eey
        elif eey is None:
            eff[n["id"]] = fsy
        else:
            eff[n["id"]] = min(fsy, eey)
    return eff


def build_snapshot(data: dict, cutoff_year: int) -> dict:
    eff_first_seen = compute_eff_first_seen(data)
    nodes = [n for n in data["nodes"]
             if eff_first_seen.get(n["id"]) is not None and eff_first_seen[n["id"]] <= cutoff_year]
    node_ids = {n["id"] for n in nodes}

    edges = []
    dropped_partial = 0
    for e in data["hyperedges"]:
        edge_year = e.get("provenance", {}).get("article_year")
        if edge_year is None or edge_year > cutoff_year:
            continue
        members = e["members"]
        if all(m in node_ids for m in members):
            edges.append(e)
        else:
            dropped_partial += 1

    return {
        "cutoff_year": cutoff_year,
        "nodes": nodes,
        "edges": edges,
        "dropped_partial_edges": dropped_partial,
    }


def describe_snapshot(snap: dict) -> dict:
    nodes, edges = snap["nodes"], snap["edges"]
    type_counts = Counter(n["type"] for n in nodes)
    relation_counts = Counter(e["relation_type"] for e in edges)
    arities = [len(e["members"]) for e in edges]

    # Degree = number of incident hyperedges *within this snapshot's edge
    # set*. This matters for the clustering method (Deliverable 0): if most
    # nodes have degree 1, the node-assignment rule is trivial for them, and
    # we want that visible from T1 rather than discovered later.
    degree = Counter()
    for e in edges:
        for m in e["members"]:
            degree[m] += 1
    node_ids = {n["id"] for n in nodes}
    degrees = [degree.get(nid, 0) for nid in node_ids]
    n_nodes = len(nodes)
    singleton_fraction = round(sum(1 for d in degrees if d == 1) / n_nodes, 3) if n_nodes else None
    isolated_fraction = round(sum(1 for d in degrees if d == 0) / n_nodes, 3) if n_nodes else None

    return {
        "cutoff_year": snap["cutoff_year"],
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "node_type_distribution": dict(type_counts.most_common()),
        "relation_type_distribution": dict(relation_counts.most_common()),
        "arity_min": min(arities) if arities else None,
        "arity_max": max(arities) if arities else None,
        "arity_mean": round(sum(arities) / len(arities), 2) if arities else None,
        "arity_distribution_top10": dict(Counter(arities).most_common(10)),
        "degree_min": min(degrees) if degrees else None,
        "degree_max": max(degrees) if degrees else None,
        "degree_mean": round(sum(degrees) / len(degrees), 2) if degrees else None,
        "singleton_fraction": singleton_fraction,
        "isolated_fraction": isolated_fraction,
        "dropped_partial_edges": snap["dropped_partial_edges"],
    }


def describe_growth(prev_desc: dict, curr_desc: dict) -> dict:
    return {
        "from_year": prev_desc["cutoff_year"],
        "to_year": curr_desc["cutoff_year"],
        "new_nodes": curr_desc["n_nodes"] - prev_desc["n_nodes"],
        "new_edges": curr_desc["n_edges"] - prev_desc["n_edges"],
        "node_growth_pct": round(
            100 * (curr_desc["n_nodes"] - prev_desc["n_nodes"]) / prev_desc["n_nodes"], 1
        ) if prev_desc["n_nodes"] else None,
        "edge_growth_pct": round(
            100 * (curr_desc["n_edges"] - prev_desc["n_edges"]) / prev_desc["n_edges"], 1
        ) if prev_desc["n_edges"] else None,
    }


def describe_growth_by_type(prev_snap: dict, curr_snap: dict) -> dict:
    """Break growth down by node type / relation type ('churn'), since
    snapshots are cumulative and net totals alone hide which parts of the
    graph are actually driving growth between two cutoffs."""
    prev_node_types = Counter(n["type"] for n in prev_snap["nodes"])
    curr_node_types = Counter(n["type"] for n in curr_snap["nodes"])
    node_delta = {
        t: curr_node_types.get(t, 0) - prev_node_types.get(t, 0)
        for t in curr_node_types
    }

    prev_rel_types = Counter(e["relation_type"] for e in prev_snap["edges"])
    curr_rel_types = Counter(e["relation_type"] for e in curr_snap["edges"])
    edge_delta = {
        r: curr_rel_types.get(r, 0) - prev_rel_types.get(r, 0)
        for r in curr_rel_types
    }

    return {
        "from_year": prev_snap["cutoff_year"],
        "to_year": curr_snap["cutoff_year"],
        "new_nodes_by_type": {k: v for k, v in sorted(node_delta.items(), key=lambda kv: -kv[1]) if v},
        "new_edges_by_relation_type": {k: v for k, v in sorted(edge_delta.items(), key=lambda kv: -kv[1]) if v},
    }


def check_data_quality(data: dict) -> list:
    """A few basic sanity checks; extend as more issues are found."""
    issues = []

    n_nodes = len(data["nodes"])
    n_origin_year = sum(1 for n in data["nodes"] if n.get("origin_year") is not None)
    issues.append(
        f"Only {n_origin_year}/{n_nodes} nodes have a true `origin_year` "
        f"(the rest fall back to `first_seen_year`); treat `year` as a "
        f"'first appeared in this corpus' proxy, not a verified invention date."
    )

    all_node_ids = {n["id"] for n in data["nodes"]}
    dangling = 0
    for e in data["hyperedges"]:
        if not all(m in all_node_ids for m in e["members"]):
            dangling += 1
    if dangling:
        issues.append(f"{dangling}/{len(data['hyperedges'])} hyper-edges reference at least one node id not present in the node list.")

    no_year_edges = sum(1 for e in data["hyperedges"] if e.get("provenance", {}).get("article_year") is None)
    if no_year_edges:
        issues.append(f"{no_year_edges}/{len(data['hyperedges'])} hyper-edges have no `provenance.article_year` and will be excluded from all snapshots.")

    # Corrected temporal-leak check (fix found by external review): using
    # `year`/`origin_year` for snapshot membership let entities into a
    # snapshot before the corpus itself had any record of them. Quantify
    # how many nodes this affected, per cutoff, as a permanent record of
    # the bug's real size rather than fixing it silently.
    leak_counts = {}
    for cutoff in (2020, 2022, 2024):
        leaked = sum(
            1 for n in data["nodes"]
            if n.get("year") is not None and n["year"] <= cutoff
            and n.get("first_seen_year") is not None and n["first_seen_year"] > cutoff
        )
        leak_counts[cutoff] = leaked
    issues.append(
        f"TEMPORAL LEAK, FOUND AND FIXED: filtering snapshots on `year` "
        f"(falls back to `origin_year`) instead of `first_seen_year` let "
        f"nodes into a snapshot before the corpus had any record of them — "
        f"{leak_counts[2020]} nodes at cutoff=2020, {leak_counts[2022]} at "
        f"2022, {leak_counts[2024]} at 2024. Snapshots are now built on "
        f"`first_seen_year` (nodes) and `provenance.article_year` (edges); "
        f"see the module docstring."
    )

    node_first_seen = {n["id"]: n["first_seen_year"] for n in data["nodes"] if n.get("first_seen_year") is not None}
    inconsistent = 0
    for e in data["hyperedges"]:
        member_years = [node_first_seen[m] for m in e["members"] if m in node_first_seen]
        edge_year = e.get("provenance", {}).get("article_year")
        if member_years and edge_year is not None and edge_year < max(member_years):
            inconsistent += 1
    if inconsistent:
        issues.append(
            f"{inconsistent}/{len(data['hyperedges'])} hyper-edges have a "
            f"`provenance.article_year` earlier than the max `first_seen_year` "
            f"among their member nodes. This is expected and handled correctly: "
            f"it is exactly why 'dropped_partial_edges' occurs at intermediate "
            f"cutoffs even though the edge's own article_year is <= cutoff."
        )

    # Duplicate surface forms: the same normalized surface_form appearing
    # under more than one distinct node id, *regardless of node type*
    # (e.g. "graph neural networks" recorded separately as both a
    # `technique` and a `method` node). These are likely the same
    # real-world entity recorded under separate ids, which would
    # artificially split a single concept across hyperedges/clusters if
    # left unmerged.
    surface_groups = {}
    for n in data["nodes"]:
        sf = n.get("surface_form")
        if not sf:
            continue
        key = sf.strip().lower()
        surface_groups.setdefault(key, set()).add(n["id"])
    duplicate_groups = {k: v for k, v in surface_groups.items() if len(v) > 1}
    if duplicate_groups:
        example = sorted(duplicate_groups.items(), key=lambda kv: -len(kv[1]))[0]
        issues.append(
            f"{len(duplicate_groups)} distinct surface forms map to more than "
            f"one node id across the whole dataset "
            f"({sum(len(v) for v in duplicate_groups.values())} node ids total), "
            f"suggesting duplicate entities not merged at export time (e.g. the "
            f"same technique recorded once per article, or once per type). "
            f"Worst case: {example[0]!r} spans {len(example[1])} separate ids. "
            f"These are NOT merged automatically; the clustering method may "
            f"need a de-duplication pass, or this should be treated as a known "
            f"limitation of the raw export."
        )

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tkh_collection10.json")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=DEFAULT_CUTOFFS)
    parser.add_argument("--out", default="outputs/t1_snapshot_stats.json")
    args = parser.parse_args()

    data = load_tkh(args.data)

    print(f"Loaded {len(data['nodes'])} nodes and {len(data['hyperedges'])} hyper-edges "
          f"from {args.data}\n")

    descriptions = []
    snapshots_raw = []
    for cy in sorted(args.cutoffs):
        snap = build_snapshot(data, cy)
        snapshots_raw.append(snap)
        desc = describe_snapshot(snap)
        descriptions.append(desc)
        print(f"--- Snapshot <= {cy} ---")
        print(f"  nodes: {desc['n_nodes']}, edges: {desc['n_edges']}")
        print(f"  node types: {desc['node_type_distribution']}")
        print(f"  arity range: {desc['arity_min']}-{desc['arity_max']} "
              f"(mean {desc['arity_mean']})")
        print(f"  degree range: {desc['degree_min']}-{desc['degree_max']} "
              f"(mean {desc['degree_mean']}), singleton fraction: "
              f"{desc['singleton_fraction']}")
        if desc["dropped_partial_edges"]:
            print(f"  [!] {desc['dropped_partial_edges']} edges dropped "
                  f"(reference nodes outside this snapshot)")
        print()

    growth = []
    growth_by_type = []
    for prev_snap, curr_snap, prev, curr in zip(
        snapshots_raw[:-1], snapshots_raw[1:], descriptions[:-1], descriptions[1:]
    ):
        g = describe_growth(prev, curr)
        growth.append(g)
        gt = describe_growth_by_type(prev_snap, curr_snap)
        growth_by_type.append(gt)
        print(f"Growth {g['from_year']} -> {g['to_year']}: "
              f"+{g['new_nodes']} nodes ({g['node_growth_pct']}%), "
              f"+{g['new_edges']} edges ({g['edge_growth_pct']}%)")
        print(f"  by node type: {gt['new_nodes_by_type']}")
        print(f"  by relation type: {gt['new_edges_by_relation_type']}")

    print("\n--- Data quality notes ---")
    issues = check_data_quality(data)
    for i in issues:
        print(f"  - {i}")

    Path("outputs").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "snapshots": descriptions,
            "growth": growth,
            "growth_by_type": growth_by_type,
            "data_quality_notes": issues,
        }, f, indent=2)
    print(f"\nSaved stats to {args.out}")


if __name__ == "__main__":
    main()
