# Literature Review

This review covers the areas required by the task: (1) hierarchical/
multi-resolution community detection, (2) hypergraph clustering / spectral
hypergraph partitioning, (3) graph coarsening, and (4) dynamic/evolutionary
community detection. It also documents two additional papers found during a
deeper search for the method's actual algorithmic foundation, since they turn
out to be closer to our setting than any of the four required-area papers
alone. Every paper here was checked against its primary source (arXiv
abstract, publisher DOI, or PubMed/PMC record) before being cited; no
fabricated or misattributed papers are used.

## 1. Hierarchical / multi-resolution community detection

**Dreveton, Kuroda, Grossglauser, Thiran (2026).** *When Does Bottom-Up Beat
Top-Down in Hierarchical Community Detection?* Journal of the American
Statistical Association, 121(554), 1284-1295.

Compares agglomerative (bottom-up) and divisive (top-down) strategies for
building a hierarchy of communities, with recovery guarantees under a
Hierarchical Stochastic Block Model. Operates on pairwise graphs, not
hypergraphs.

**What we borrow:** the framing of hierarchy construction as a choice between
two structurally different strategies (bottom-up vs. top-down), each with its
own failure modes. This directly informs our own justification in Deliverable
0 for why we build our hierarchy bottom-up (see paper 5 below).

**What we do differently:** their method assumes a pairwise graph and no
temporal dimension. Our setting requires both a hyperedge-native structure
(P4) and cross-snapshot stability (P5), neither of which this paper addresses.

## 2. Hypergraph clustering / spectral hypergraph partitioning

**Ni, Zeng, Mu, Lin (2026).** *From Representation to Clusters: A Contrastive
Learning Approach for Attributed Hypergraph Clustering* (CAHC). Proceedings
of the ACM Web Conference 2026 (WWW), 1080-1091.

Jointly learns node representations and cluster assignments on attributed
hypergraphs, using both node-level and hyperedge-level contrastive
objectives, so cluster assignment is informed by both hypergraph structure
and node content/attributes.

**What we borrow:** the core idea of explicitly combining a structural signal
(hyperedge-level) and a content/attribute signal (node-level) rather than
picking one. This is the closest existing analogue to our own P3
reconciliation problem (structure vs. semantics).

**What we do differently:** CAHC produces a single flat clustering, not a
laminar multi-level hierarchy (P1, P2), and it is evaluated on a single
static snapshot, with no temporal coupling across snapshots (P5).

## 3. Hypergraph coarsening

**Aghdaei, Zhao, Feng (2021).** *HyperSF: Spectral Hypergraph Coarsening via
Flow-based Local Clustering.* IEEE/ACM International Conference on
Computer-Aided Design (ICCAD).

Proposes a spectral hypergraph coarsening scheme that aggregates vertices
while preserving the structural/spectral properties of the original
hypergraph, using flow-based local clustering and spectral clustering on the
corresponding bipartite graph.

Note on recency: this paper falls outside the suggested 2022-2026 window
named in the task. It was kept over newer alternatives (e.g. HySpecPro, 2026,
which is a single-level partitioner rather than a coarsening method) because
it directly addresses hyperedge aggregation during coarsening, which is
exactly the problem central to Task T4. We treat this as a deliberate
trade-off between recency and topical precision, not an oversight.

**What we borrow:** the general principle that a hyperedge should not be
collapsed by a single fixed rule regardless of how its endpoints are
distributed across coarser clusters; the coarsening rule should depend on how
much of the hyperedge remains internal to one cluster.

**What we do differently:** HyperSF's coarsening rule is designed to preserve
spectral/structural properties for downstream partitioning quality, not to
preserve human-interpretable semantic boundaries or track identity across
temporal snapshots. Our T4 rule instead has to serve P4 fidelity and stay
consistent with P6 (labels must not overclaim what remains of a collapsed
edge).

## 4. Dynamic / evolutionary community detection

**Vusirikkayala, Viswanatham (2026).** *TSA-HGNN: A Stability-Aware
Multi-Scale Temporal Graph Neural Network for Dynamic Community Detection.*
Frontiers in Artificial Intelligence, 9.

Despite "HGNN" in the name, this is a temporal graph neural network operating
on ordinary pairwise graph snapshots (hypergraph methods appear only as
comparison baselines). It combines GraphSAGE snapshot embeddings with
short-term (TCN) and long-term (Informer-style) temporal modeling, plus an
explicit stability objective to reduce spurious community changes between
snapshots.

**What we borrow:** the principle of adding an explicit stability objective
rather than relying on a clustering algorithm to be "accidentally" stable
across reruns. This directly motivates our approach to P5.

**What we do differently:** TSA-HGNN operates on pairwise graphs and produces
a flat (non-hierarchical) community assignment per snapshot. We need
stability at every level of a laminar hierarchy simultaneously, and our
notion of stability must also support explicit birth/growth/merge/split/death
event tracking, which TSA-HGNN does not provide.

## 5. Core method foundation: hypergraph edge clustering

The four papers above cover the four required areas, but none of them
directly matches our setting (hypergraph + hierarchical + temporal at once).
A further search for a more direct algorithmic foundation for our own method
turned up two closely related papers, which we use as the actual basis of
our clustering method rather than just background reading.

**Lotito, Musciotto, Montresor, Battiston (2023).** *Hyperlink communities in
higher-order networks.* (arXiv:2303.01385; related to Communications
Physics / PNAS-adjacent work in the same research group.)

Instead of clustering nodes, this paper clusters the hyperedges themselves:
it defines a Jaccard distance between hyperedges (based on shared member
nodes), builds a full distance matrix, and runs single-linkage hierarchical
clustering to obtain a dendrogram. Cutting the dendrogram at different
heights gives community structures at different scales, and since each
hyperedge is placed in exactly one community, but nodes inherit the
communities of every hyperedge they belong to, nodes can end up in more than
one community. The paper explicitly shows (their Fig. 2) that this avoids a
key failure mode of clique-expansion: a single large hyperedge added to the
hypergraph can turn into a dense clique under projection, which distorts or
destroys the hierarchical structure a projection-based method would find.
Open-source code is available as part of the `hypergraphx` library.

**DeWolfe, Théberge (2025).** *Detecting Patterns of Interaction in Temporal
Hypergraphs via Edge Clustering.* arXiv:2506.03105.

Extends the same edge-clustering idea to temporal hypergraphs. Two hyperedges
are considered similar only if they are both structurally similar (Jaccard
over members) and close in time, combined as
`w(i,j) = sqrt(s_Jaccard(i,j) * T_sigma(i,j))`, where `T_sigma` is a linear
time-decay kernel with a tunable width `sigma`. Clustering is again done via
single-linkage on the resulting weighted line graph, but cluster extraction
uses HDBSCAN's excess-of-mass method instead of a single fixed cut height,
which allows clusters of different "heights" (strengths) to be selected
simultaneously and leaves weak/ambiguous hyperedges as outliers rather than
forcing them into a cluster. Demonstrated on a 1.2M-hyperedge real-world
collaboration hypergraph.

**What we borrow from these two papers together:** the core idea of
clustering hyperedges directly (using set-overlap between their members)
rather than projecting to a pairwise graph, which is a genuinely
hypergraph-native operation; the resulting dendrogram as a natural multi-
resolution structure; and the specific idea of multiplying a structural
similarity by a temporal-decay kernel to make the clustering time-aware.

**What we do differently, and why we cannot use either method as-is:**

1. Both papers allow **overlapping** node communities (a node inherits every
   community of every hyperedge it participates in). The task's P1 explicitly
   rules overlapping clusters out of scope. We therefore add a **hard
   assignment step**: after cutting the dendrogram, each node is assigned to
   the single cluster containing the largest share of its incident
   hyperedges (ties broken by total incident hyperedge weight). We treat
   "keep the overlap, as in the original papers" as the credible alternative
   we considered and rejected, specifically because it conflicts with P1 as
   stated in this task, not because overlap is a bad idea in general.
2. Neither paper enforces an exact community-count budget; DeWolfe &
   Théberge's HDBSCAN-based cut does not target a specific number of
   clusters at all. We still need a level with 10-15 communities (P2), so we
   search over cut heights (or an equivalent HDBSCAN parameter) for one that
   lands in that range, rather than accepting whatever the default
   extraction method returns.
3. Neither paper incorporates a semantic/content signal (P3); both distances
   are purely structural (Jaccard over shared members, optionally scaled by
   time). We add a semantic term based on embedding similarity between
   hyperedges' member surface forms, combined with the structural and
   temporal terms.
4. DeWolfe & Théberge's time kernel is a **continuous sliding window**
   (parameterized by `sigma`), whereas our task defines **discrete cumulative
   snapshots**. We adapt the idea by using it to compare a hyperedge's
   position in the current snapshot against its own history in prior
   snapshots (for T3 identity tracking) rather than as a continuous decay
   over the whole timeline.

## Positioning summary

No single existing paper combines a hyperedge-native structure, a genuine
multi-level (not just multi-cut) hierarchy respecting a strict top-level size
budget, an explicit structure/semantics trade-off, and real cross-snapshot
identity tracking. Papers 1 and 4 handle temporal or hierarchical structure
but only on pairwise graphs; papers 2 and 3 handle hypergraphs but only at a
single static snapshot; papers 5 (Lotito et al.; DeWolfe & Théberge) come
closest, hypergraph-native and (in the second case) temporally aware, but
allow overlapping communities and have no semantic signal or exact size
budget. Our method is built directly on top of paper 5's edge-clustering
approach, modified to satisfy P1 (hard assignment) and P2 (budget-targeted
cut selection), and extended with a semantic term (borrowing the
structure+content combination principle from CAHC), an explicit stability
objective (in the spirit of TSA-HGNN), and a coarsening rule for hyperedges
(in the spirit of HyperSF's overlap-dependent treatment) for Task T4.

## Verification notes

All papers were checked against primary sources before being cited here
(arXiv abstracts, publisher DOIs, or PubMed/PMC records), not accepted on the
basis of a search snippet alone.

- During the initial four-paper search, a second independent AI check
  flagged the TSA-HGNN paper as a possible fabrication (no result found via
  a quick author/title search). We re-verified it directly against PubMed
  (PMID 42290696, PMCID PMC13261176, DOI 10.3389/frai.2026.1824901) and
  confirmed it is a real, indexed publication (Frontiers in Artificial
  Intelligence, vol. 9, May 2026). The false positive likely occurred
  because the paper is very recent and may not yet be indexed everywhere.
- Papers 5 (Lotito et al., 2023; DeWolfe & Théberge, 2025) were each fetched
  and read in full directly from arXiv (2303.01385 and 2506.03105
  respectively) before being described here, rather than relying on search
  snippets. The similarity/weight formulas and the HDBSCAN-based cluster
  extraction detail quoted above were confirmed against the papers' own
  method sections, and Lotito et al.'s open-source implementation was
  confirmed to exist as part of the `hypergraphx` library referenced in
  their paper.
