# Report

## Deliverable 0 — Formal Problem Statement

### Method summary

Our method builds the multi-resolution hierarchy by clustering **hyperedges**,
not nodes, then deriving a node-level hierarchy from the resulting hyperedge
hierarchy. This choice follows directly from the literature review (see
`literature_review.md`, section 5): hyperedge clustering via a dendrogram
(Lotito et al., 2023) is the most direct existing hypergraph-native technique
that produces a natural multi-level structure, and it avoids the information
loss of projecting the hypergraph to a pairwise graph before clustering.

Concretely, for a given snapshot with hyperedge set `E`:

1. **Pairwise hyperedge similarity.** For every pair of hyperedges `i, j`,
   compute:
   - a **structural** similarity `s_struct(i,j)`, the Jaccard index of their
     member node sets: `|M(i) ∩ M(j)| / |M(i) ∪ M(j)|`.
   - a **semantic** similarity `s_sem(i,j)`, the cosine similarity between
     sentence embeddings of the concatenated surface forms of each
     hyperedge's members.
   - a combined similarity: `sim(i,j) = s_struct(i,j)^α · s_sem(i,j)^(1-α)`,
     a weighted geometric mean, with `α ∈ [0,1]` a single tunable parameter.
     Distance is `d(i,j) = 1 - sim(i,j)`.
2. **Hierarchical agglomerative clustering (HAC)** on the resulting distance
   matrix (average-linkage), producing a full dendrogram over hyperedges.
3. **Level extraction.** Cut the dendrogram at multiple heights to obtain a
   sequence of hyperedge partitions `Q_0 (coarsest) ... Q_K (finest, one
   hyperedge per cluster)`. For `Q_0`, search over candidate cut heights for
   the one producing a cluster count closest to, and within, the 10-15 range
   required by P2.
4. **Node assignment (top-down, laminar by construction).** At the coarsest
   level, assign each node to the `Q_0` cluster containing the largest share
   of its incident hyperedges. At every subsequent, finer level, restrict the
   candidate clusters to the **descendants of the node's already-assigned
   parent cluster** (i.e. only clusters that emerged from splitting that
   parent), and again assign to whichever descendant holds the largest share
   of the node's incident hyperedges. This top-down, restricted assignment
   is what guarantees P1 for nodes (see "Guarantees" below); computing each
   level's assignment independently would not.

### Objective / criterion

The method does not optimize a single global objective function end-to-end
(no joint loss is minimized). Instead, it satisfies a **criterion**: at every
level, the reported partition is the one produced by cutting a hyperedge
dendrogram built from a fixed similarity function, subject to the level-0
size constraint (P2) and the top-down node-assignment rule (P1). This is a
deliberate choice: an explicit two-stage/deterministic procedure (build
dendrogram, then extract levels, then assign nodes) rather than a single
joint optimization.

### Structure/semantics trade-off (P3)

The trade-off is made explicit and controllable through the single parameter
`α` in `sim(i,j) = s_struct(i,j)^α · s_sem(i,j)^(1-α)`. `α = 1` reduces to a
purely structural method (closest to Lotito et al.'s original formulation);
`α = 0` reduces to purely semantic clustering. We do not hard-code a single
value of `α` as correct; instead (per T6) we treat `α` as an ablation
variable and report how coherence, stability, and downstream utility change
across a small sweep of values, then report which value we would ship and
why.

**Alternative considered and rejected:** a joint contrastive-learning
objective in the style of CAHC (Ni et al., 2026), which learns a combined
representation rather than combining two precomputed similarity scores. We
rejected this for our setting because (a) contrastive learning requires
enough data to avoid overfitting, and our corpus (~5,800 nodes, ~1,400
hyperedges at full scale, fewer at early snapshots) is small for that; (b) a
learned representation is harder to re-verify by hand, which matters for the
verification habits this task asks us to demonstrate; and (c) the explicit
`α` parameter is easier to sweep, report, and justify than an implicit
learned trade-off.

### Complexity

Let `m = |E|` be the number of hyperedges in a snapshot. The distance matrix
requires `O(m²)` pairwise similarity computations; HAC with average linkage
is `O(m² log m)` with a standard priority-queue implementation. For our
largest snapshot (`m ≈ 1,429`), this is on the order of 2 million pairwise
computations, which is well within the "minutes, not hours" budget the task
specifies. Node assignment is `O(m · d̄)` where `d̄` is the average node degree
(number of incident hyperedges), negligible by comparison. This method would
not scale to `m` in the hundreds of thousands without approximation (e.g.
locality-sensitive hashing to avoid the full `O(m²)` matrix), but that is
explicitly out of scope per the task's data-size guidance.

### What is guaranteed by construction vs. what is empirical

| Property | Status | Why |
|---|---|---|
| **P1** (laminar refinement) | **Guaranteed by construction** | Dendrogram cuts at different heights are always nested (a standard property of HAC); the top-down, descendant-restricted node-assignment rule inherits this nesting rather than computing each level independently. |
| **P2** (size budget, 10-15 at level 0) | **Best-effort, not strictly guaranteed** | A dendrogram may not have a cut height producing exactly 10-15 clusters (a single high-level merge can jump the count, e.g. from 9 directly to 16). We search for the closest achievable cut and report the actual count achieved; if it falls outside the range, this is stated honestly rather than hidden. |
| **P3** (semantic coherence) | **Structurally guaranteed to consider both signals; the resulting trade-off quality is empirical** | The similarity function always incorporates both terms; whether a given `α` produces clusters a domain expert would recognize as coherent is evaluated in T6, not assumed. |
| **P4** (hyperedge fidelity) | **Guaranteed by construction** | Clustering operates on hyperedges via a member-overlap distance; no pairwise clique-expansion projection is used to drive clustering (a projection is only built separately, if at all, for the T2-required comparison of what it would lose). |
| **P5** (temporal stability) | **Partially structural, largely empirical** | Cross-snapshot cluster identity is established by matching, not by re-deriving it from scratch with no reference to the past (see T3 for the exact matching mechanism); the resulting stability under real growth and under perturbation is measured, not assumed. |
| **P6** (faithful labels) | **Empirical** | Labels are LLM-generated; faithfulness is checked post-hoc via an over-claim rate (T6), not guaranteed by the generation process itself. |

### Note on scope

This document will be extended with the T3 temporal-matching mechanism, the
T4 hyperedge-collapse rule, and full T6 results as those tasks are
completed. This section (Deliverable 0) is considered stable; later
additions will not change the method described above without an explicit
changelog note.
