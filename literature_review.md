# Literature Review

This review covers four papers spanning the areas required by the task:
(1) hierarchical/multi-resolution community detection, (2) hypergraph
clustering / spectral hypergraph partitioning, (3) graph coarsening, and
(4) dynamic/evolutionary community detection. Each paper was independently
verified against its primary source (arXiv, publisher DOI) before inclusion;
no fabricated or misattributed papers are cited here.

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
0 for why we chose one direction over the other.

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
static snapshot, with no temporal coupling across snapshots (P5). We adapt
the structure+content combination idea into a hierarchical, temporally
coupled setting.

## 3. Hypergraph coarsening

**Aghdaei, Zhao, Feng (2021).** *HyperSF: Spectral Hypergraph Coarsening via
Flow-based Local Clustering.* IEEE/ACM International Conference on
Computer-Aided Design (ICCAD).

Proposes a spectral hypergraph coarsening scheme that aggregates vertices
while preserving the structural/spectral properties of the original
hypergraph, using flow-based local clustering and spectral clustering on the
corresponding bipartite graph.

Note on recency: this paper falls outside the suggested 2022-2026 window
named in the task. It was chosen over newer alternatives (e.g. HySpecPro,
2026, which is a single-level partitioner rather than a coarsening method)
because it directly addresses hyperedge aggregation during coarsening, which
is exactly the problem central to Task T4. We treat this as a deliberate
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

## Positioning summary

None of the four papers combines a hyperedge-native structure with real
temporal/dynamic coupling; papers 1 and 4 handle temporal or hierarchical
structure but only on pairwise graphs, while papers 2 and 3 handle
hypergraphs but only at a single, static snapshot. This gap between
"hypergraph-native" and "temporally stable" methods is exactly where this
task's method sits, and is the main reason no single prior method could be
adapted wholesale; our method instead combines ideas from across all four
papers (structure/content reconciliation from CAHC, coarsening-by-overlap
from HyperSF, an explicit stability objective from TSA-HGNN, and an explicit
hierarchy-direction justification in the style of Dreveton et al.).

## Verification note

All four papers were checked against primary sources before being cited
here (arXiv abstracts, publisher DOIs, or PubMed/PMC records), not accepted
on the basis of a search snippet alone. During this process, a second
independent check flagged the TSA-HGNN paper as a possible fabrication (no
result found via a quick author/title search). We re-verified it directly
against PubMed (PMID 42290696, PMCID PMC13261176, DOI
10.3389/frai.2026.1824901) and confirmed it is a real, indexed publication
(Frontiers in Artificial Intelligence, vol. 9, May 2026). The false positive
likely occurred because the paper is very recent and may not yet be indexed
everywhere. We keep this note as a record of the verification step, per the
task's request to document what was checked and how.
