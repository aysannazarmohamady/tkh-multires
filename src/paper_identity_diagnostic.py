"""
T2 diagnostic — paper-identity NMI, reported against a null model, plus a
count of clusters dominated by a single article's edges.

NOT a target metric (an external review correctly pointed out the earlier
"NMI < 0.3" target had no real justification — removed). This is reported
as a diagnostic: how much does the clustering structure correlate with
"which paper" versus a null model with the same cluster-size distribution
but no real structure.

Usage:
    python src/paper_identity_diagnostic.py --hierarchy outputs/hierarchy.json
"""

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def compute_diagnostic(hierarchy_path, data_path, level=0, n_null=1000, seed=0):
    data = json.load(open(data_path))
    h = json.load(open(hierarchy_path))

    node_article = {}
    for n in data["nodes"]:
        arts = n.get("provenance", {}).get("articles")
        if arts:
            node_article[n["id"]] = arts[0]

    assignment = h["node_assignment"]
    common = [nid for nid in assignment if nid in node_article and assignment[nid][level] is not None]
    labels = np.array([assignment[nid][level] for nid in common])
    articles = np.array([node_article[nid] for nid in common])

    observed_nmi = normalized_mutual_info_score(labels, articles)

    rng = np.random.default_rng(seed)
    null_nmis = []
    for _ in range(n_null):
        shuffled = rng.permutation(labels)  # preserves cluster-size distribution exactly
        null_nmis.append(normalized_mutual_info_score(shuffled, articles))
    null_nmis = np.array(null_nmis)

    # Clusters where >50% of the clustering-eligible edges come from one article
    edge_ids = h.get("edge_ids")
    edge_labels_all_levels = h.get("edge_cluster_labels")
    single_article_clusters, total_clusters = None, None
    if edge_ids and edge_labels_all_levels:
        edge_labels = edge_labels_all_levels[level]
        # need edge -> article_id; reload from data hyperedges (match by id, including merged ids)
        edge_article = {}
        for e in data["hyperedges"]:
            edge_article[e["id"]] = e.get("provenance", {}).get("article_id")
        # merged evaluated_on edges aren't in data["hyperedges"], but their
        # provenance came from the first group edge (see merge_evaluated_on_by_article);
        # approximate by leaving them out if not found.
        cluster_edges = {}
        for eid, lbl in zip(edge_ids, edge_labels):
            cluster_edges.setdefault(lbl, []).append(edge_article.get(eid))
        dominant_fracs = {}
        for lbl, arts in cluster_edges.items():
            arts = [a for a in arts if a is not None]
            if not arts:
                continue
            top_count = Counter(arts).most_common(1)[0][1]
            dominant_fracs[lbl] = top_count / len(arts)
        single_article_clusters = sum(1 for f in dominant_fracs.values() if f > 0.5)
        total_clusters = len(dominant_fracs)

    return {
        "level": level,
        "n_nodes_with_article": len(common),
        "observed_nmi": round(float(observed_nmi), 4),
        "null_nmi_mean": round(float(null_nmis.mean()), 4),
        "null_nmi_std": round(float(null_nmis.std()), 4),
        "null_nmi_95ci": [round(float(x), 4) for x in np.percentile(null_nmis, [2.5, 97.5])],
        "z_vs_null": round(float((observed_nmi - null_nmis.mean()) / null_nmis.std()), 2) if null_nmis.std() > 0 else None,
        "clusters_with_single_article_majority_over_50pct": single_article_clusters,
        "total_clusters_at_level": total_clusters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", default="outputs/hierarchy.json")
    parser.add_argument("--data", default="data/tkh_collection10.json")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", default="outputs/paper_identity_diagnostic.json")
    args = parser.parse_args()

    result = compute_diagnostic(args.hierarchy, args.data, args.level)
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
