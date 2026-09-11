"""
T6 — Extrinsic utility: does the hierarchy help a coarse-to-fine retrieval
task, compared to a flat baseline and a null (shape-preserving, shuffled)
hierarchy?

Design (per external review):
- Scorer: TF-IDF cosine similarity (a genuinely different embedding family
  from the MiniLM sentence-transformer used for clustering, so the scorer
  doesn't share the clustering's geometry). Used for BOTH the hierarchical
  system's cluster/node representations and the flat baseline's node
  representations, and for scoring each question's query text.
- Label-free: cluster representation is the mean TF-IDF vector of its
  members' surface forms. This makes the extrinsic result independent of
  T5 (which is not yet run at full scale).
- Hierarchical system: beam search — score level-0 clusters, keep top-b,
  descend into their children, re-score, repeat to the leaf level, then
  rank the member nodes of visited leaves.
- Flat baseline: rank ALL candidate nodes directly with the same scorer.
- Null hierarchy: same cluster sizes at every level, but member ID's
  shuffled across leaves at random (same shape, no real structure). If the
  real hierarchy doesn't beat this null, the hierarchy adds nothing.
- Cost = number of node/super-node texts scored. Report recall@cost curves
  and steps-to-first-hit, not a single hit-rate number.
- Duplicate target ids: a target name is one equivalence class; a hit is
  counted if ANY of its ids is retrieved.
- Q5, Q11 (zero resolvable targets) excluded from the score, reported as
  coverage failures. Q15-18 (claims task, not methods) out of scope.
- Only the 2026 snapshot is used ("by Feb 2026" in each question).

Usage:
    python src/extrinsic_eval.py
"""

import json
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BEAM_WIDTH = 3


def normalize(s):
    s = s.strip().lower()
    s = re.sub(r"[\-_/]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_everything():
    data = json.load(open("data/tkh_collection10.json"))
    hier = json.load(open("outputs/hierarchy.json"))  # 2026 snapshot only
    gt = json.load(open("data/ground_truth.json"))
    import csv
    questions = {row["question_id"]: row["question"]
                 for row in csv.DictReader(open("data/questions.csv"), delimiter=";")}
    return data, hier, gt, questions


def resolve_targets(gt, data):
    surface_to_ids = defaultdict(list)
    for n in data["nodes"]:
        surface_to_ids[normalize(n["surface_form"])].append(n["id"])

    node_type = {n["id"]: n["type"] for n in data["nodes"]}
    per_question_targets = {}
    for qid, entry in gt.items():
        ems = entry.get("expected_methods", [])
        if not ems:
            continue
        resolved = {}
        for name in ems:
            ids = surface_to_ids.get(normalize(name), [])
            resolved[name] = {"ids": ids, "types": sorted(set(node_type[i] for i in ids))}
        per_question_targets[qid] = resolved
    return per_question_targets


def build_tfidf_space(nodes):
    docs = [n["surface_form"] for n in nodes]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    node_ids = [n["id"] for n in nodes]
    return vectorizer, matrix, node_ids


def cluster_mean_vectors(node_ids, matrix, cluster_of):
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    groups = defaultdict(list)
    for nid, cid in cluster_of.items():
        if nid in idx_of:
            groups[cid].append(idx_of[nid])
    means = {}
    for cid, idxs in groups.items():
        means[cid] = np.asarray(matrix[idxs].mean(axis=0))
    return means


def score(query_vec, candidate_matrix):
    return cosine_similarity(query_vec, candidate_matrix).flatten()


def rank_of(scores: np.ndarray, idx: int):
    """Fractional rank of candidate `idx` within `scores` (1 = best).
    Fix (found by external review): a zero TF-IDF score means the query
    and candidate share NO vocabulary at all — not a weak match, but no
    signal whatsoever. An earlier version ranked zero-score candidates by
    whatever order a Python `set` happened to iterate them in (which
    depends on hash randomization, verified externally to shift a
    "hit" position by ~100 depending on PYTHONHASHSEED). Zero-score
    candidates are now explicitly excluded (`None`, not retrieved); among
    nonzero scores, ties get the standard fractional/mid rank."""
    s = scores[idx]
    if s <= 0:
        return None
    return float((scores > s).sum() + ((scores == s).sum() + 1) / 2)


def beam_search_ranking(query_vec, hier, node_ids, matrix, n_levels):
    """Returns (overhead_cost, {node_id: fractional_rank_or_None within the
    final visited-leaf candidate set})."""
    assignment = hier["node_assignment"]
    overhead = 0

    cluster_of_l0 = {nid: v[0] for nid, v in assignment.items() if v[0] is not None}
    means = cluster_mean_vectors(node_ids, matrix, cluster_of_l0)
    cluster_ids = sorted(means.keys())  # deterministic order
    cluster_matrix = np.vstack([means[c] for c in cluster_ids])
    scores = score(query_vec, cluster_matrix)
    overhead += len(cluster_ids)
    order = np.argsort(-scores)
    active = set(cluster_ids[i] for i in order[:BEAM_WIDTH])
    active_nodes = set(nid for nid, c in cluster_of_l0.items() if c in active)

    for level in range(1, n_levels):
        cluster_of_lvl = {nid: v[level] for nid, v in assignment.items()
                           if v[level] is not None and nid in active_nodes}
        if not cluster_of_lvl:
            break
        means = cluster_mean_vectors(node_ids, matrix, cluster_of_lvl)
        cluster_ids = sorted(means.keys())
        if not cluster_ids:
            break
        cluster_matrix = np.vstack([means[c] for c in cluster_ids])
        scores = score(query_vec, cluster_matrix)
        overhead += len(cluster_ids)
        order = np.argsort(-scores)
        active = set(cluster_ids[i] for i in order[:BEAM_WIDTH])
        active_nodes = set(nid for nid, c in cluster_of_lvl.items() if c in active)

    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    candidate_ids = sorted(nid for nid in active_nodes if nid in idx_of)  # deterministic
    if not candidate_ids:
        return overhead, {}
    candidate_idxs = [idx_of[nid] for nid in candidate_ids]
    candidate_matrix = matrix[candidate_idxs]
    scores = score(query_vec, candidate_matrix)
    ranks = {nid: rank_of(scores, i) for i, nid in enumerate(candidate_ids)}
    return overhead, ranks


def flat_ranking_ranks(query_vec, node_ids, matrix):
    """Same tie-aware rank_of logic, applied to the full candidate set."""
    scores = score(query_vec, matrix)
    ranks = {nid: rank_of(scores, i) for i, nid in enumerate(node_ids)}
    return ranks


def build_null_hierarchy(hier, seed):
    rng = np.random.default_rng(seed)
    assignment = hier["node_assignment"]
    node_ids = [nid for nid, v in assignment.items() if v[0] is not None]

    shuffled_ids = list(node_ids)
    rng.shuffle(shuffled_ids)
    id_map = dict(zip(node_ids, shuffled_ids))

    null_assignment = {}
    for real_nid in node_ids:
        donor = id_map[real_nid]
        null_assignment[real_nid] = assignment[donor]
    return {"node_assignment": null_assignment}


def hit_position(ranks: dict, target_id_sets: list, overhead: int = 0):
    """For each target (a set of equivalent node ids), the best (lowest)
    fractional rank among its ids, plus overhead — or None if none of its
    ids have a nonzero score (not retrieved at all, not just far down the
    ranking)."""
    positions = []
    for id_set in target_id_sets:
        candidate_ranks = [ranks[nid] for nid in id_set if nid in ranks and ranks[nid] is not None]
        positions.append(overhead + min(candidate_ranks) if candidate_ranks else None)
    return positions


def main():
    data, hier, gt, questions = load_everything()
    per_q_targets = resolve_targets(gt, data)

    nodes_2026 = data["nodes"]
    vectorizer, matrix, node_ids = build_tfidf_space(nodes_2026)
    n_levels = len(next(iter(hier["node_assignment"].values())))

    usable_questions, coverage_failures = [], []
    for qid, targets in per_q_targets.items():
        resolvable = {name: info for name, info in targets.items() if info["ids"]}
        (usable_questions if resolvable else coverage_failures).append(qid)

    print(f"Usable questions ({len(usable_questions)}): {usable_questions}")
    print(f"Coverage failures ({len(coverage_failures)}): {coverage_failures}")

    n_null_seeds = 5
    per_question_detail = {}
    total_targets = 0
    total_found_h, total_found_f = 0, 0

    for qid in usable_questions:
        query_vec = vectorizer.transform([questions[qid]])
        targets = per_q_targets[qid]
        target_id_sets = [set(info["ids"]) for info in targets.values() if info["ids"]]
        target_types = {name: info["types"] for name, info in targets.items() if info["ids"]}
        total_targets += len(target_id_sets)

        overhead_h, ranks_h = beam_search_ranking(query_vec, hier, node_ids, matrix, n_levels)
        pos_h = hit_position(ranks_h, target_id_sets, overhead=overhead_h)

        ranks_f = flat_ranking_ranks(query_vec, node_ids, matrix)
        pos_f = hit_position(ranks_f, target_id_sets, overhead=0)

        null_positions = []
        for seed in range(n_null_seeds):
            null_hier = build_null_hierarchy(hier, seed)
            overhead_n, ranks_n = beam_search_ranking(query_vec, null_hier, node_ids, matrix, n_levels)
            null_positions.append(hit_position(ranks_n, target_id_sets, overhead=overhead_n))

        total_found_h += sum(1 for p in pos_h if p is not None)
        total_found_f += sum(1 for p in pos_f if p is not None)

        per_question_detail[qid] = {
            "n_targets": len(target_id_sets),
            "hierarchical_hit_positions": pos_h, "hierarchical_overhead": overhead_h,
            "flat_hit_positions": pos_f,
            "null_hierarchy_hit_positions_per_seed": null_positions,
            "target_types": target_types,
        }
        print(f"{qid}: n_targets={len(target_id_sets)} "
              f"h_found={sum(1 for p in pos_h if p)} flat_found={sum(1 for p in pos_f if p)}")

    print(f"\n=== Deterministic, tie-aware result (zero-similarity = not retrieved) ===")
    print(f"Total targets: {total_targets}")
    print(f"Hierarchical found (any cost): {total_found_h}/{total_targets}")
    print(f"Flat found (any cost): {total_found_f}/{total_targets}")
    print(f"Interpretation: with TF-IDF as the scorer, most targets have ZERO lexical "
          f"overlap with the natural-language question text and are therefore never "
          f"retrieved by either system, regardless of cost. This is a scorer limitation, "
          f"not evidence the hierarchy is uninformative — but it means this specific "
          f"result is UNINFORMATIVE about the hierarchy's value, and should be reported "
          f"as a null/inconclusive result rather than a comparison, until re-run with a "
          f"denser (non-lexical) scorer.")

    with open("outputs/extrinsic_eval_detail.json", "w") as f:
        json.dump({
            "usable_questions": usable_questions,
            "coverage_failures": coverage_failures,
            "beam_width": BEAM_WIDTH,
            "per_question": per_question_detail,
        }, f, indent=2)

    with open("outputs/extrinsic_eval_summary.json", "w") as f:
        json.dump({
            "usable_questions": usable_questions,
            "coverage_failures": coverage_failures,
            "total_targets": total_targets,
            "hierarchical_found_any_cost": total_found_h,
            "flat_found_any_cost": total_found_f,
            "conclusion": "NULL / UNINFORMATIVE RESULT: TF-IDF gives zero lexical similarity "
                          "between natural-language questions and short technical target names "
                          "for the large majority of targets, so neither system can retrieve them "
                          "regardless of cost. This is a scorer limitation, not a finding about "
                          "the hierarchy's usefulness. A denser embedding scorer is needed before "
                          "this comparison is informative.",
            "caveat": "An earlier version of this script reported a positive-looking result "
                      "(hierarchical beating flat at low cost) that was an artifact of "
                      "non-deterministic Python set iteration order breaking ties among "
                      "zero-similarity candidates — corrected here.",
        }, f, indent=2)
    print("\nSaved outputs/extrinsic_eval_detail.json and outputs/extrinsic_eval_summary.json")


if __name__ == "__main__":
    main()
