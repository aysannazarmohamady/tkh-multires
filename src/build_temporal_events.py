"""
T3 — Temporal coupling: persistent cluster identity + event log.

MECHANISM DECISION (revised after a deeper external review): temporal
regularization (lambda > 0) was tested and REJECTED. Two independent
findings ruled it out:
1. Perturbation-robustness test (10% edge removal, 5 seeds): lambda=0 has
   the HIGHEST mean perturbation-ARI (0.457) of any value tested
   (lambda_sd in {0, 0.1, 0.25, 0.5} of the distance distribution's std);
   every nonzero lambda tested was worse. See
   `outputs/lambda_selection_and_null.json`.
2. A circularity problem in how "stability" was being measured: the
   regularizer subtracts lambda from the distance of hyperedge pairs that
   were co-clustered in the PREVIOUS snapshot, and stability was then
   measured as ARI against that same previous snapshot's partition. The
   mechanism was optimizing the exact metric used to evaluate it, so a
   large apparent "improvement" (transition ARI 0.42 -> 0.81 in an earlier,
   buggy version) could not fail to appear regardless of whether it
   reflected anything real. We caught and neutralized this ourselves before
   it reached the final report — see report.md, "T3 circularity found and
   fixed" for the full account, including a null model (shuffled prior
   labels) that confirms lambda=0 behaves identically whether the prior is
   real or shuffled, exactly as it should since lambda=0 never reads the
   prior at all.

This module therefore uses `lambda_sd=0` (no regularization): clusters are
identified purely by post-hoc hyperedge-Jaccard matching across
independently-clustered snapshots. Identity is TRACKED, not artificially
enforced. Real transition ARI (0.43-0.53, mean 0.49, with the corrected
rank-normalized similarity) is reported honestly as the actual level of
cross-snapshot consistency, not inflated by a circular mechanism.

Matching signal: hyperedge-set Jaccard overlap between a snapshot-t level-0
cluster and a snapshot-(t+1) level-0 cluster, restricted to hyperedges
present in both snapshots (an edge born at t+1 cannot be "in" any t-cluster).
"""

import json
from collections import defaultdict

import numpy as np

from temporal_reg import run_chain, CUTOFFS

CONTINUATION_THRESHOLD = 0.5  # mutual best match above this -> continuation
BIRTH_THRESHOLD = 0.1         # below this (or no match) -> counted as new


def cluster_edge_sets(edge_labels: dict) -> dict:
    """{cluster_id: set(edge_id)} from {edge_id: cluster_id}."""
    out = defaultdict(set)
    for eid, cid in edge_labels.items():
        out[cid].add(eid)
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


def main():
    data = json.load(open("data/tkh_collection10.json"))
    emb_path = "outputs/semantic_embeddings.npy"
    order_path = "outputs/embedding_edge_order.json"
    alpha, lam_sd = 0.5, 0.0

    print(f"Running chain (alpha={alpha}, lambda_sd={lam_sd} — no regularization, "
          f"evidence-based choice; see module docstring)...")
    chain = run_chain(data, alpha, lam_sd, emb_path, order_path, cutoffs=CUTOFFS)

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

    with open("outputs/temporal_events.json", "w") as f:
        json.dump({
            "reliability_note": (
                "Temporal regularization (lambda > 0) was tested and rejected: "
                "it did not improve perturbation-robustness (lambda=0 had the "
                "highest mean perturbation-ARI of any value tested), and an "
                "earlier version's large apparent transition-ARI gain turned out "
                "to be circular (the regularizer directly manipulates the same "
                "metric used to evaluate it). This log uses lambda=0: identity "
                "is tracked via post-hoc hyperedge-Jaccard matching only, not "
                "artificially enforced. See outputs/lambda_selection_and_null.json "
                "and report.md ('T3 circularity found and fixed') for the full "
                "evidence trail. Real cross-snapshot ARI (~0.43-0.53, mean 0.49) "
                "is the honest reliability level for events below — some "
                "'continued' classifications may still reflect an imperfect "
                "matching threshold rather than deep conceptual continuity."
            ),
            "level": 0,
            "alpha": alpha,
            "lambda_sd": lam_sd,
            "continuation_threshold": CONTINUATION_THRESHOLD,
            "birth_threshold": BIRTH_THRESHOLD,
            "events_by_snapshot": all_events,
        }, f, indent=2)
    print("\nSaved outputs/temporal_events.json")


if __name__ == "__main__":
    main()
