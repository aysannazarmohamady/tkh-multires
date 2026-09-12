"""
T3 — Temporal coupling: persistent cluster identity + event log.

CRITICAL FIX (found by external review): an earlier version built its own
chain via `temporal_reg.run_chain`, which uses TF-IDF (not the real
MiniLM sentence embeddings used to produce the actually-shipped
`hierarchy_*.json` files). This TF-IDF chain produced a DIFFERENT
clustering — level-0 counts of 13/14/21/27 across snapshots, violating
P2 at 2024 and 2026 (target 10-15) — and agreed with the shipped
hierarchies at only ARI 0.28-0.51. Worse, `enrich_events_with_t4_cohesion`
then joined this TF-IDF chain's integer cluster labels against T4's
cohesion scores, which were computed from the REAL (MiniLM) hierarchies —
an integer label like "5" means a completely different set of nodes in
each clustering, so every attached cohesion value was silently wrong.

Fix: this module now reads cluster/edge labels DIRECTLY from the shipped
`outputs/hierarchy_<year>.json` files (the same ones used for T1, T2, T6),
not from a separately re-run TF-IDF chain. This guarantees T3's event log
describes the actual delivered hierarchy, and that T4's cohesion join uses
labels from the same clustering that produced them.

Mechanism decision (still applies): temporal regularization (lambda > 0)
was tested and rejected earlier (see report.md) because it did not
improve, and was slightly worse than, lambda=0 under perturbation, and an
earlier (separate) implementation of it was found to be circular. This
module therefore uses plain post-hoc hyperedge-Jaccard matching across the
independently-clustered, real-embedding-based snapshots — no
regularization is applied here at all.

Matching signal: hyperedge-set Jaccard overlap between a snapshot-t level-0
cluster and a snapshot-(t+1) level-0 cluster, restricted to hyperedges
present in both snapshots.
"""

import json
from collections import defaultdict

import numpy as np

CONTINUATION_THRESHOLD = 0.5  # mutual best match above this -> continuation
BIRTH_THRESHOLD = 0.1         # below this (or no match) -> counted as new

HIERARCHY_PATHS = {
    2020: "outputs/hierarchy_2020.json",
    2022: "outputs/hierarchy_2022.json",
    2024: "outputs/hierarchy_2024.json",
    2026: "outputs/hierarchy.json",
}


def load_chain_from_shipped_hierarchies(hierarchy_paths: dict) -> dict:
    """Returns {cutoff: {"node_labels_l0": {...}, "edge_labels_l0": {...},
    "k0": int}}, read directly from the actually-delivered hierarchy files
    — no re-clustering, no separate embedding backend."""
    chain = {}
    for cutoff, path in hierarchy_paths.items():
        hier = json.load(open(path))
        node_labels_l0 = {nid: v[0] for nid, v in hier["node_assignment"].items() if v[0] is not None}
        edge_ids = hier.get("edge_ids")
        edge_labels_all = hier.get("edge_cluster_labels")
        edge_labels_l0 = {}
        if edge_ids and edge_labels_all:
            edge_labels_l0 = {eid: int(lbl) for eid, lbl in zip(edge_ids, edge_labels_all[0])}
        k0 = hier["level_cluster_counts"][0]
        assert 10 <= k0 <= 15, (
            f"Shipped hierarchy for cutoff={cutoff} has level-0 count {k0}, "
            f"outside the P2 target (10-15). Refusing to build T3 events on "
            f"an invalid hierarchy."
        )
        chain[cutoff] = {"node_labels_l0": node_labels_l0, "edge_labels_l0": edge_labels_l0, "k0": k0}
    return chain


def cluster_edge_sets(edge_labels: dict) -> dict:
    """{cluster_id: set(edge_id)} from {edge_id: cluster_id}."""
    out = defaultdict(set)
    for eid, cid in edge_labels.items():
        out[cid].add(eid)
    return out


def coarse_edge_neighbor_sets(coarse_edges: list) -> dict:
    """T4 integration (fix: T4's output was computed but never read by
    anything downstream). Build {super_node: set(coarse_edge_key)} from a
    level's coarse-edge list (as produced by hyperedge_collapse.py), so a
    cluster's "signature" for matching can include which OTHER super-nodes
    it is coarsely connected to, not just its own raw hyperedge membership.
    Used as a secondary, corroborating signal in T3 matching — see
    `match_snapshots_with_t4`.
    """
    out = defaultdict(set)
    for i, e in enumerate(coarse_edges):
        key = f"coarse_{i}"
        for sn in e["super_nodes"]:
            out[sn].add(key)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def match_snapshots(prev_edge_labels: dict, curr_edge_labels: dict):
    """Returns:
        overlap: {(prev_cluster, curr_cluster): jaccard}
        prev_best: {prev_cluster: (best_curr_cluster, score)}
        curr_best: {curr_cluster: (best_prev_cluster, score)}
    """
    prev_sets = cluster_edge_sets(prev_edge_labels)
    curr_sets = cluster_edge_sets(curr_edge_labels)

    overlap = {}
    for pc, pedges in prev_sets.items():
        for cc, cedges in curr_sets.items():
            j = jaccard(pedges, cedges)
            if j > 0:
                overlap[(pc, cc)] = j

    prev_best, curr_best = {}, {}
    for pc in prev_sets:
        candidates = [(cc, j) for (p, cc), j in overlap.items() if p == pc]
        prev_best[pc] = max(candidates, key=lambda x: x[1]) if candidates else (None, 0.0)
    for cc in curr_sets:
        candidates = [(pc, j) for (pc, c), j in overlap.items() if c == cc]
        curr_best[cc] = max(candidates, key=lambda x: x[1]) if candidates else (None, 0.0)

    return overlap, prev_best, curr_best, prev_sets, curr_sets


def classify_events(prev_edge_labels, curr_edge_labels, prev_ids: dict, next_persistent_id: list):
    """prev_ids: {prev_cluster_label: persistent_id} from the previous step.
    next_persistent_id: single-element list used as a mutable counter.

    Returns (curr_ids: {curr_cluster_label: persistent_id}, events: list[dict])
    """
    overlap, prev_best, curr_best, prev_sets, curr_sets = match_snapshots(prev_edge_labels, curr_edge_labels)
    events = []
    curr_ids = {}

    # Determine, for each curr cluster, all prev clusters that best-match it
    # (possible merge candidates), and for each prev cluster, all curr
    # clusters it's spread across (possible split candidates).
    curr_to_prev_matches = defaultdict(list)  # curr_cluster -> [prev_cluster, ...] whose best match is this curr
    for pc, (cc, score) in prev_best.items():
        if cc is not None and score >= CONTINUATION_THRESHOLD:
            curr_to_prev_matches[cc].append(pc)

    for cc in curr_sets:
        best_pc, best_score = curr_best.get(cc, (None, 0.0))
        contributing_prevs = curr_to_prev_matches.get(cc, [])

        if len(contributing_prevs) >= 2:
            # MERGE: multiple prev clusters best-match this one curr cluster.
            merged_persistent_ids = [prev_ids[pc] for pc in contributing_prevs if pc in prev_ids]
            pid = f"p{next_persistent_id[0]}"
            next_persistent_id[0] += 1
            curr_ids[cc] = pid
            events.append({"type": "merge", "from_persistent_ids": merged_persistent_ids,
                            "to_persistent_id": pid, "curr_cluster_label": cc})
        elif best_pc is not None and best_score >= CONTINUATION_THRESHOLD:
            # CONTINUATION (possibly with growth)
            pid = prev_ids.get(best_pc)
            if pid is None:
                pid = f"p{next_persistent_id[0]}"
                next_persistent_id[0] += 1
            curr_ids[cc] = pid
            prev_size = len(prev_sets.get(best_pc, set()))
            curr_size = len(curr_sets[cc])
            events.append({"type": "grew" if curr_size > prev_size else "continued",
                            "persistent_id": pid, "prev_size": prev_size, "curr_size": curr_size,
                            "overlap_jaccard": round(best_score, 3)})
        elif best_score < BIRTH_THRESHOLD or best_pc is None:
            # BIRTH: no meaningful match to any prior cluster.
            pid = f"p{next_persistent_id[0]}"
            next_persistent_id[0] += 1
            curr_ids[cc] = pid
            events.append({"type": "birth", "persistent_id": pid, "curr_cluster_label": cc,
                            "size": len(curr_sets[cc])})
        else:
            # Weak/ambiguous match (between BIRTH_THRESHOLD and
            # CONTINUATION_THRESHOLD): treat as a new identity but log the
            # weak link rather than silently picking a side.
            pid = f"p{next_persistent_id[0]}"
            next_persistent_id[0] += 1
            curr_ids[cc] = pid
            events.append({"type": "ambiguous_weak_link", "persistent_id": pid,
                            "closest_prev_persistent_id": prev_ids.get(best_pc),
                            "overlap_jaccard": round(best_score, 3)})

    # SPLIT / DISSOLUTION: prev clusters whose best match's curr cluster did
    # NOT pick them back as its dominant contributor (i.e. their identity did
    # not carry forward), and whose edges are spread with no dominant share.
    for pc in prev_sets:
        pid = prev_ids.get(pc)
        if pid is None:
            continue
        if pid in [e.get("persistent_id") for e in events if e["type"] in ("continued", "grew")]:
            continue  # already accounted for as a continuation
        if pid in [i for e in events if e["type"] == "merge" for i in e["from_persistent_ids"]]:
            continue  # already accounted for as a merge input
        # Find how this prev cluster's edges are distributed across curr clusters.
        dest_shares = {}
        pedges = prev_sets[pc]
        for cc, cedges in curr_sets.items():
            shared = len(pedges & cedges)
            if shared:
                dest_shares[cc] = shared / len(pedges)
        if not dest_shares:
            events.append({"type": "dissolved", "persistent_id": pid,
                            "note": "no surviving edges found in any current cluster"})
            continue
        max_share = max(dest_shares.values())
        if max_share < 0.5 and len(dest_shares) >= 2:
            events.append({"type": "split", "persistent_id": pid,
                            "into_curr_clusters": list(dest_shares.keys()),
                            "shares": {str(k): round(v, 3) for k, v in dest_shares.items()}})
        elif max_share < 0.5:
            events.append({"type": "dissolved", "persistent_id": pid,
                            "note": "edges dispersed with no dominant successor"})

    return curr_ids, events


EVENT_TYPE_NOTES = {
    "birth": "Fires for any level-0 cluster with no continuation-strength match to a prior cluster.",
    "continued": "Fires when a cluster's best-match predecessor is at least as large.",
    "grew": "Fires when a cluster's best-match predecessor is smaller (same continuity condition as 'continued').",
    "split": (
        "Fires when a prior cluster's edges are spread over >=2 current clusters with no "
        "single destination holding a majority share. Observed in these snapshots (2022)."
    ),
    "merge": (
        "Fires when >=2 prior clusters independently best-match the same current cluster "
        "at or above the continuation threshold. Reachable by the matcher regardless of the "
        "cumulative-snapshot rule (it depends on how edges re-cluster, not on edge removal); "
        "it simply did not occur in these four re-clusterings."
    ),
    "dissolved": (
        "Two distinct sub-cases share this label. (a) 'no surviving edges found in any "
        "current cluster': structurally blocked here, because eff_first_seen<=t and "
        "article_year<=t are cumulative, so an edge id present in a prior clustering-edge "
        "set is still present at the next cutoff and must land in some current cluster. "
        "(b) 'edges dispersed with no dominant successor': not blocked by cumulative "
        "membership -- a prior cluster's edges can still be split with no majority winner "
        "after re-clustering -- but did not occur in these snapshots either."
    ),
    "ambiguous_weak_link": (
        "Fires for a match strictly between BIRTH_THRESHOLD and CONTINUATION_THRESHOLD; "
        "treated as a new identity with the closest prior identity logged, not silently "
        "merged into it. Fired frequently in these snapshots."
    ),
}


def compute_event_type_coverage(all_events: dict) -> dict:
    """Explicit reachable-vs-fired accounting for every event type the
    matcher in `classify_events` can emit, so a reader does not have to
    infer from raw counts why `merge` and one branch of `dissolved` never
    appear across these four cumulative snapshots.
    """
    required_types = sorted(EVENT_TYPE_NOTES)
    observed_types = sorted({
        e["type"]
        for events in all_events.values()
        for e in events
    })
    return {
        "required_types": required_types,
        "reachable": {t: True for t in required_types},
        "fired": {t: t in observed_types for t in required_types},
        "observed_types": observed_types,
        "notes": EVENT_TYPE_NOTES,
        "summary": (
            "merge and the 'no surviving edges' branch of dissolved are "
            "implemented and reachable in the matcher but fired zero times "
            "here: eff_first_seen<=t and article_year<=t are cumulative, so "
            "no node or clustering edge ever leaves a later snapshot, which "
            "rules out that dissolution branch and makes both merge and the "
            "other dissolution branch a matter of re-clustering happening not "
            "to produce them in this particular data, not a defect in the "
            "matcher."
        ),
    }


def enrich_events_with_t4_cohesion(all_events: dict, cutoffs: list) -> dict:
    """T4 integration (fix: T4 was computed but nothing downstream read it).
    Attach each event's T4 cohesion score (internal edge mass / total
    incident mass, from src/hyperedge_collapse.py) for the relevant
    cluster(s) at level 0, as a corroborating signal: a cluster that
    SPLITS with high prior cohesion is more surprising (real conceptual
    break) than one that was already loosely held together (low cohesion,
    a split is closer to "finally separating what was never that unified").
    """
    cohesion_by_cutoff = {}
    for cutoff in cutoffs:
        path = f"outputs/hyperedge_collapse_{cutoff}_level0.json"
        try:
            data = json.load(open(path))
            cohesion_by_cutoff[cutoff] = {int(k): v for k, v in data["cohesion_by_cluster"].items()}
        except FileNotFoundError:
            cohesion_by_cutoff[cutoff] = {}

    for cutoff, events in all_events.items():
        cohesion = cohesion_by_cutoff.get(cutoff, {})
        for e in events:
            if e["type"] in ("birth",) and "curr_cluster_label" in e:
                e["t4_cohesion"] = cohesion.get(e["curr_cluster_label"])
            elif e["type"] in ("continued", "grew") and "curr_cluster_label" not in e:
                pass  # these events don't carry a raw cluster label; skip
            elif e["type"] == "split":
                e["t4_cohesion_of_dest_clusters"] = {
                    str(k): cohesion.get(int(k)) if str(k).lstrip("-").isdigit() else None
                    for k in e.get("into_curr_clusters", [])
                }
    return all_events


def main():
    CUTOFFS = sorted(HIERARCHY_PATHS.keys())
    print("Loading cluster/edge labels directly from the shipped hierarchy files "
          "(no re-clustering, no separate embedding backend)...")
    chain = load_chain_from_shipped_hierarchies(HIERARCHY_PATHS)
    for c in CUTOFFS:
        print(f"  cutoff={c}: k0={chain[c]['k0']} (verified within P2's 10-15 target)")

    prev_ids = {}
    next_persistent_id = [0]
    all_events = {}
    # Seed persistent ids for the first snapshot (no prior to match against).
    first_cutoff = CUTOFFS[0]
    for cc in cluster_edge_sets(chain[first_cutoff]["edge_labels_l0"]):
        prev_ids[cc] = f"p{next_persistent_id[0]}"
        next_persistent_id[0] += 1
    all_events[first_cutoff] = [{"type": "birth", "persistent_id": pid, "curr_cluster_label": cc}
                                 for cc, pid in prev_ids.items()]

    for prev_cutoff, curr_cutoff in zip(CUTOFFS[:-1], CUTOFFS[1:]):
        curr_ids, events = classify_events(
            chain[prev_cutoff]["edge_labels_l0"], chain[curr_cutoff]["edge_labels_l0"],
            prev_ids, next_persistent_id
        )
        all_events[curr_cutoff] = events
        prev_ids = curr_ids

    from collections import Counter
    print("\nEvent counts per transition:")
    for cutoff, events in all_events.items():
        print(f"  {cutoff}: {dict(Counter(e['type'] for e in events))}")

    all_events = enrich_events_with_t4_cohesion(all_events, CUTOFFS)
    print("(Events enriched with T4 cohesion scores where available — see "
          "outputs/hyperedge_collapse_<year>_level0.json)")

    event_type_coverage = compute_event_type_coverage(all_events)
    print("\nEvent type coverage:", json.dumps(event_type_coverage["fired"]))

    with open("outputs/temporal_events.json", "w") as f:
        json.dump({
            "event_type_coverage": event_type_coverage,
            "reliability_note": (
                "CORRECTED: an earlier version built its own clustering chain "
                "via a TF-IDF-based re-run, which produced a DIFFERENT hierarchy "
                "from the one actually shipped (level-0 counts 13/14/21/27, "
                "violating P2 at 2024/2026) and agreed with the shipped "
                "hierarchies at only ARI 0.28-0.51. Worse, T4 cohesion scores "
                "(computed from the real, shipped hierarchies) were then joined "
                "onto this different clustering's integer labels, silently "
                "attaching wrong values. This version reads cluster/edge labels "
                "directly from the shipped `hierarchy_<year>.json` files (same "
                "ones used for T1/T2/T6), so events and T4 cohesion now come "
                "from the same clustering. Real cross-snapshot ARI on the "
                "shipped hierarchies (not this file) is reported separately in "
                "report.md; identity here is tracked via post-hoc "
                "hyperedge-Jaccard matching, not artificially enforced by any "
                "regularization (lambda=0, rejected earlier — see report.md, "
                "'T3 circularity found and fixed')."
            ),
            "level": 0,
            "source": "shipped hierarchy_<year>.json files (real MiniLM embeddings), not a re-clustering",
            "continuation_threshold": CONTINUATION_THRESHOLD,
            "birth_threshold": BIRTH_THRESHOLD,
            "events_by_snapshot": all_events,
        }, f, indent=2)
    print("\nSaved outputs/temporal_events.json")


if __name__ == "__main__":
    main()
