"""
T1 — Load and describe the evolving TKH graph.

Cumulative (not disjoint) snapshots are used, since the task requires a
sequence of graph states over time, not independent yearly slices — later
snapshots must contain earlier ones for growth/stability comparisons in T3
to make sense.

An edge is only included once ALL of its member nodes are present in the
snapshot. We considered including an edge whenever its own `year` <= cutoff
regardless of its members, but that would let a snapshot reference nodes it
doesn't otherwise contain — inconsistent with treating each snapshot as a
self-contained hypergraph. This choice is what surfaces `dropped_partial_edges`,
which turned out to flag a real data issue (see check_data_quality): edge
years and node years don't always agree.

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


def build_snapshot(data: dict, cutoff_year: int) -> dict:
    """Return the subset of nodes/edges valid at or before cutoff_year."""
    nodes = [n for n in data["nodes"] if n.get("year") is not None and n["year"] <= cutoff_year]
    node_ids = {n["id"] for n in nodes}

    edges = []
    dropped_partial = 0
    for e in data["hyperedges"]:
        if e.get("year") is None or e["year"] > cutoff_year:
            continue
        members = e["members"]
        if all(m in node_ids for m in members):
            edges.append(e)
        else:
            # Edge's own year <= cutoff, but references a node that appears
            # later (e.g. missing/inconsistent node year). Data-quality flag.
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
    arity_counts = Counter(len(e["members"]) for e in edges)
    relation_counts = Counter(e["relation_type"] for e in edges)

    arities = [len(e["members"]) for e in edges]
    return {
        "cutoff_year": snap["cutoff_year"],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "node_type_distribution": dict(type_counts.most_common()),
        "relation_type_distribution": dict(relation_counts.most_common()),
        "arity_min": min(arities) if arities else None,
        "arity_max": max(arities) if arities else None,
        "arity_mean": round(sum(arities) / len(arities), 2) if arities else None,
        "arity_distribution_top10": dict(Counter(arities).most_common(10)),
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

    no_year_edges = sum(1 for e in data["hyperedges"] if e.get("year") is None)
    if no_year_edges:
        issues.append(f"{no_year_edges}/{len(data['hyperedges'])} hyper-edges have no `year` and will be excluded from all snapshots.")

    node_years = {n["id"]: n["year"] for n in data["nodes"] if n.get("year") is not None}
    inconsistent = 0
    for e in data["hyperedges"]:
        member_years = [node_years[m] for m in e["members"] if m in node_years]
        if member_years and e.get("year") is not None and e["year"] < max(member_years):
            inconsistent += 1
    if inconsistent:
        issues.append(
            f"{inconsistent}/{len(data['hyperedges'])} hyper-edges have a `year` "
            f"earlier than the max `year` among their member nodes — edge.year and "
            f"node.year likely come from different definitions (e.g. article_year vs "
            f"first_seen_year) and are not mutually consistent. This is why some "
            f"'dropped_partial_edges' occur at intermediate cutoffs even though the "
            f"edge's own year is <= cutoff."
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
    for cy in sorted(args.cutoffs):
        snap = build_snapshot(data, cy)
        desc = describe_snapshot(snap)
        descriptions.append(desc)
        print(f"--- Snapshot <= {cy} ---")
        print(f"  nodes: {desc['n_nodes']}, edges: {desc['n_edges']}")
        print(f"  node types: {desc['node_type_distribution']}")
        print(f"  arity range: {desc['arity_min']}-{desc['arity_max']} "
              f"(mean {desc['arity_mean']})")
        if desc["dropped_partial_edges"]:
            print(f"  [!] {desc['dropped_partial_edges']} edges dropped "
                  f"(reference nodes outside this snapshot)")
        print()

    growth = []
    for prev, curr in zip(descriptions[:-1], descriptions[1:]):
        g = describe_growth(prev, curr)
        growth.append(g)
        print(f"Growth {g['from_year']} -> {g['to_year']}: "
              f"+{g['new_nodes']} nodes ({g['node_growth_pct']}%), "
              f"+{g['new_edges']} edges ({g['edge_growth_pct']}%)")

    print("\n--- Data quality notes ---")
    issues = check_data_quality(data)
    for i in issues:
        print(f"  - {i}")

    Path("outputs").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "snapshots": descriptions,
            "growth": growth,
            "data_quality_notes": issues,
        }, f, indent=2)
    print(f"\nSaved stats to {args.out}")


if __name__ == "__main__":
    main()
