"""
T4 — Hyper-edge collapse: coarsening operator on the incidence structure.

Design (from external review, adopted after verification): for a hyperedge
e with endpoints spread across super-nodes at level k, sigma(e) = the set
of distinct super-nodes its members land in, with multiplicity m_S = number
of e's endpoints landing in super-node S:

- |sigma(e)| == 1 (all endpoints in one super-node): e becomes INTERNAL.
  Not deleted — recorded in that super-node's internal_edges, contributing
  to a cohesion score (internal edge mass / total incident mass) usable for
  T5/T6.
- |sigma(e)| == 2: a coarse binary edge between the two super-nodes, weight
  1/(|sigma(e)|-1) = 1 (recovers ordinary graph coarsening at arity 2),
  carrying the multiplicity vector (m_A, m_B) so partial internalization
  isn't lost.
- |sigma(e)| >= 3: kept as an actual HYPEREDGE of arity |sigma(e)| between
  those super-nodes (not clique-expanded into pairwise edges). This is
  what makes the coarsening hypergraph-native rather than a projection.
  Implemented behind `--clique-expand` for direct comparison: running both
  and diffing lets us quantify what a pairwise projection would have lost,
  which is what the task explicitly asks for when a projection is tried.

Coarse edge weight: w(E_coarse) = sum over contributing original edges e of
1/(|sigma(e)|-1). Relation type and min/max year of contributing edges are
retained (needed for T3 event attribution and P6 temporal honesty: a
coarsened edge must not claim a time range wider than its actual sources).

What this loses (stated per the design, not hidden): which specific members
grounded the relation, and the distinction between e.g. a (1,1,8) endpoint
spread and a (3,3,4) spread landing in the same 3 super-nodes. The
multiplicity vector is stored specifically so this loss is at least
recoverable, not silently discarded.
"""

import argparse
import json
from collections import defaultdict


def load_hierarchy_and_data(hierarchy_path, data_path, level=0):
    hier = json.load(open(hierarchy_path))
    data = json.load(open(data_path))
    node_to_cluster = {nid: v[level] for nid, v in hier["node_assignment"].items() if v[level] is not None}
    return hier, data, node_to_cluster


def collapse_hyperedges(all_edges, node_to_cluster, clique_expand=False):
    """Returns (internal_by_cluster, coarse_edges).

    internal_by_cluster: {cluster_id: {"internal_edges": [...], "internal_mass": int}}
    coarse_edges: list of coarsened edges (arity-2 or, if clique_expand is
    False, arity>=3 hyperedges between super-nodes; if clique_expand is
    True, arity>=3 cases are expanded into all pairwise combinations, for
    direct comparison of what a projection would lose).
    """
    internal_by_cluster = defaultdict(lambda: {"internal_edges": [], "internal_mass": 0})
    coarse_edge_agg = {}  # key: frozenset(super-nodes) or tuple for clique pairs -> aggregated record

    total_incident_mass = defaultdict(int)

    for e in all_edges:
        members = e["members"]
        clusters_of_members = [node_to_cluster.get(m) for m in members]
        if any(c is None for c in clusters_of_members):
            continue  # a member has no cluster assignment at this level; skip (should be rare after T2's post-hoc attachment)

        multiplicity = defaultdict(int)
        for c in clusters_of_members:
            multiplicity[c] += 1
        sigma = set(multiplicity.keys())

        for c in multiplicity:
            total_incident_mass[c] += multiplicity[c]

        if len(sigma) == 1:
            (only_cluster,) = tuple(sigma)
            internal_by_cluster[only_cluster]["internal_edges"].append(
                {"edge_id": e["id"], "relation_type": e["relation_type"], "arity": len(members)}
            )
            internal_by_cluster[only_cluster]["internal_mass"] += len(members)
            continue

        weight = 1.0 / (len(sigma) - 1)
        year = e.get("provenance", {}).get("article_year")

        if len(sigma) == 2 or clique_expand:
            # arity-2 coarse edges, OR clique-expanded pairs if requested
            cluster_list = sorted(sigma)
            pairs = [(cluster_list[i], cluster_list[j])
                     for i in range(len(cluster_list)) for j in range(i + 1, len(cluster_list))]
            for a, b in pairs:
                key = (a, b)
                if key not in coarse_edge_agg:
                    coarse_edge_agg[key] = {
                        "super_nodes": [a, b], "weight": 0.0, "arity": 2,
                        "relation_types": set(), "min_year": None, "max_year": None,
                        "contributing_edges": [], "multiplicities": [],
                    }
                rec = coarse_edge_agg[key]
                rec["weight"] += weight
                rec["relation_types"].add(e["relation_type"])
                rec["contributing_edges"].append(e["id"])
                rec["multiplicities"].append({a: multiplicity[a], b: multiplicity[b]})
                if year is not None:
                    rec["min_year"] = year if rec["min_year"] is None else min(rec["min_year"], year)
                    rec["max_year"] = year if rec["max_year"] is None else max(rec["max_year"], year)
        else:
            # |sigma(e)| >= 3, kept as a genuine coarsened hyperedge (not clique-expanded)
            key = frozenset(sigma)
            if key not in coarse_edge_agg:
                coarse_edge_agg[key] = {
                    "super_nodes": sorted(sigma), "weight": 0.0, "arity": len(sigma),
                    "relation_types": set(), "min_year": None, "max_year": None,
                    "contributing_edges": [], "multiplicities": [],
                }
            rec = coarse_edge_agg[key]
            rec["weight"] += weight
            rec["relation_types"].add(e["relation_type"])
            rec["contributing_edges"].append(e["id"])
            rec["multiplicities"].append(dict(multiplicity))
            if year is not None:
                rec["min_year"] = year if rec["min_year"] is None else min(rec["min_year"], year)
                rec["max_year"] = year if rec["max_year"] is None else max(rec["max_year"], year)

    # cohesion score per cluster: internal mass / total incident mass
    cohesion = {}
    for c in set(list(internal_by_cluster.keys()) + list(total_incident_mass.keys())):
        internal_mass = internal_by_cluster[c]["internal_mass"] if c in internal_by_cluster else 0
        total_mass = total_incident_mass.get(c, 0)
        cohesion[c] = round(internal_mass / total_mass, 3) if total_mass else None

    coarse_edges = []
    for key, rec in coarse_edge_agg.items():
        rec["relation_types"] = sorted(rec["relation_types"])
        rec["weight"] = round(rec["weight"], 4)
        coarse_edges.append(rec)

    return dict(internal_by_cluster), coarse_edges, cohesion


def run_all_levels_and_snapshots(hierarchy_paths: dict, data_path: str) -> dict:
    """Run collapse for EVERY level of EVERY snapshot, not just level 0 of
    one snapshot (fix: an earlier version ran once, and nothing downstream
    read its output — T4 was implemented but unused, i.e. P4 wasn't
    actually exercised by the method).

    hierarchy_paths: {cutoff_year: hierarchy_json_path}
    Returns {cutoff_year: {level: {"internal": ..., "coarse_native": ...,
    "coarse_clique": ..., "cohesion": ...}}}
    """
    data = json.load(open(data_path))
    results = {}
    for cutoff, hpath in hierarchy_paths.items():
        hier = json.load(open(hpath))
        n_levels = len(next(iter(hier["node_assignment"].values())))
        results[cutoff] = {}
        for level in range(n_levels):
            node_to_cluster = {nid: v[level] for nid, v in hier["node_assignment"].items() if v[level] is not None}
            internal_native, coarse_native, cohesion = collapse_hyperedges(
                data["hyperedges"], node_to_cluster, clique_expand=False
            )
            results[cutoff][level] = {
                "n_super_nodes_with_internal_edges": len(internal_native),
                "n_coarse_edges": len(coarse_native),
                "n_coarse_hyperedges_arity3plus": sum(1 for e in coarse_native if e["arity"] >= 3),
                "cohesion_by_cluster": cohesion,
                # Keep the actual coarse edges only for level 0 in the summary
                # (full detail per level is written to per-snapshot files
                # separately; this summary is for the cross-level/cross-
                # snapshot overview).
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", default="outputs/hierarchy.json")
    parser.add_argument("--data", default="data/tkh_collection10.json")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", default="outputs/hyperedge_collapse.json")
    parser.add_argument("--all", action="store_true",
                         help="Run collapse for every level of every snapshot "
                              "(2020/2022/2024/2026), not just one level of one file.")
    args = parser.parse_args()

    if args.all:
        hierarchy_paths = {
            2020: "outputs/hierarchy_2020.json",
            2022: "outputs/hierarchy_2022.json",
            2024: "outputs/hierarchy_2024.json",
            2026: "outputs/hierarchy.json",
        }
        summary = run_all_levels_and_snapshots(hierarchy_paths, args.data)
        for cutoff, levels in summary.items():
            for level, info in levels.items():
                print(f"cutoff={cutoff} level={level}: "
                      f"{info['n_coarse_edges']} coarse edges "
                      f"({info['n_coarse_hyperedges_arity3plus']} arity>=3), "
                      f"{info['n_super_nodes_with_internal_edges']} super-nodes with internal edges")

        # Also write full per-(snapshot, level) collapse detail, one file each,
        # since T3 matching and any future drill-down need the actual coarse
        # edges, not just the summary counts.
        data = json.load(open(args.data))
        for cutoff, hpath in hierarchy_paths.items():
            hier = json.load(open(hpath))
            n_levels = len(next(iter(hier["node_assignment"].values())))
            for level in range(n_levels):
                node_to_cluster = {nid: v[level] for nid, v in hier["node_assignment"].items() if v[level] is not None}
                internal_native, coarse_native, cohesion = collapse_hyperedges(
                    data["hyperedges"], node_to_cluster, clique_expand=False
                )
                out_path = f"outputs/hyperedge_collapse_{cutoff}_level{level}.json"
                with open(out_path, "w") as f:
                    json.dump({
                        "cutoff_year": cutoff, "level": level,
                        "cohesion_by_cluster": cohesion,
                        "internal_edges_by_cluster": {str(k): v for k, v in internal_native.items()},
                        "coarse_edges_native": [
                            {**e, "super_nodes": list(e["super_nodes"])} for e in coarse_native
                        ],
                    }, f, indent=2)

        with open("outputs/hyperedge_collapse_all_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved per-(snapshot,level) files (outputs/hyperedge_collapse_<year>_level<k>.json) "
              "and outputs/hyperedge_collapse_all_summary.json")
        return

    hier, data, node_to_cluster = load_hierarchy_and_data(args.hierarchy, args.data, args.level)

    internal_native, coarse_native, cohesion = collapse_hyperedges(
        data["hyperedges"], node_to_cluster, clique_expand=False
    )
    internal_clique, coarse_clique, _ = collapse_hyperedges(
        data["hyperedges"], node_to_cluster, clique_expand=True
    )

    n_native_hyperedges = sum(1 for e in coarse_native if e["arity"] >= 3)
    n_clique_pairs_from_those = sum(
        1 for e in coarse_clique if e["arity"] == 2
    ) - sum(1 for e in coarse_native if e["arity"] == 2)

    print(f"Level {args.level}: {len(internal_native)} super-nodes have internal edges")
    print(f"Native (hypergraph-preserving) coarsening: {len(coarse_native)} coarse edges "
          f"({n_native_hyperedges} of them true hyperedges of arity >= 3)")
    print(f"Clique-expanded coarsening: {len(coarse_clique)} coarse edges (all arity 2)")
    print(f"Projection loss: clique expansion turns {n_native_hyperedges} genuine "
          f"multi-way relations into {n_clique_pairs_from_those} separate pairwise "
          f"edges, discarding the fact that they co-occurred in a single relation.")

    def serialize(coarse_list):
        out = []
        for rec in coarse_list:
            r = dict(rec)
            r["super_nodes"] = list(r["super_nodes"])
            out.append(r)
        return out

    with open(args.out, "w") as f:
        json.dump({
            "level": args.level,
            "cohesion_by_cluster": cohesion,
            "internal_edges_by_cluster": {str(k): v for k, v in internal_native.items()},
            "coarse_edges_native": serialize(coarse_native),
            "coarse_edges_clique_expanded": serialize(coarse_clique),
            "projection_loss_summary": {
                "native_hyperedges_arity_3plus": n_native_hyperedges,
                "clique_expanded_pairs_replacing_them": n_clique_pairs_from_those,
            },
        }, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
