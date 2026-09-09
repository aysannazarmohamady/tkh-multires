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
   - a combined similarity: `sim(i,j) = α · s_struct(i,j) + (1-α) · s_sem(i,j)`,
     a weighted **arithmetic** mean, with `α ∈ [0,1]` a single tunable
     parameter. Distance is `d(i,j) = 1 - sim(i,j)`.

   **Correction, made before implementation.** An earlier draft of this
   document used a weighted *geometric* mean,
   `sim(i,j) = s_struct(i,j)^α · s_sem(i,j)^(1-α)`, modeled directly on the
   time-kernel formula in DeWolfe & Théberge (2025). Before writing any
   clustering code, we ran a direct check on the provided export and found
   that **only 2.03% of all hyperedge pairs (20,721 / 1,020,306) share even
   a single member node**, i.e. `s_struct(i,j) = 0` for the remaining
   97.97%. Under a geometric mean, any zero factor forces the product to
   zero for any `α > 0`, which silently deletes the semantic term precisely
   for the pairs where it would matter most (topically related hyperedges
   that happen to share no node id). We reject the geometric-mean
   combination for this reason and use the arithmetic mean above instead,
   under which `s_struct(i,j) = 0` degrades gracefully to `sim(i,j) =
   (1-α) · s_sem(i,j)` rather than annihilating the whole similarity. We
   also separately found that **87.8% of nodes (5,090 / 5,798) have degree
   1** (belong to exactly one hyperedge); we return to what this implies for
   node assignment below.
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

   **Honest note on degree-1 nodes.** 87.8% of nodes (5,090 / 5,798) belong
   to exactly one hyperedge among the clustered relation types (see below),
   so for these nodes the "largest share" computation has only one candidate
   and is trivial: the node simply inherits its single hyperedge's cluster
   at every level. This is not a defect in the rule — the node still gets a
   correct, laminar assignment — but it means the top-down restriction only
   does non-trivial work for the remaining 12.2% (~708 nodes) that have
   multiple incident hyperedges and could plausibly be pulled toward
   different clusters. The actual "intelligence" of the clustering lives in
   the hyperedge-level similarity function (every member node, including
   degree-1 ones, contributes to its hyperedge's structural and semantic
   signal); node assignment is a deterministic read-out of that, not a
   second independent decision. We report this distribution explicitly
   (T1, updated) rather than letting it go unstated.

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
`α` in `sim(i,j) = α · s_struct(i,j) + (1-α) · s_sem(i,j)`. `α = 1` reduces to
a purely structural method; `α = 0` reduces to purely semantic clustering.
Given the extreme sparsity of direct member overlap in this corpus (2.03% of
pairs nonzero; see correction above), we expect the semantic term to
dominate in practice for any `α < 1`, since the structural term is exactly
zero for the large majority of pairs. We treat this as an honest property of
this specific dataset rather than something to engineer around by default:
we do not hard-code a single value of `α` as correct; instead (per T6) we
treat `α` as an ablation variable and report how coherence, stability, and
downstream utility change across a small sweep of values, including the
degenerate ends (`α=0`, `α=1`), then report which value we would ship and
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

### Relation-type selection: clustering set vs. held-out coherence probe

Two more issues surfaced from a direct audit of the export before writing any
code, and both are resolved by the same decision, made now rather than left
for T6.

**Issue A — degenerate edges dominating the distance matrix.** `claims` is
the single largest relation type (690 / 1,429 edges, 48.3% of all
hyperedges), with mean arity 2.69 — near-minimal, mostly linking one
`article`/`method` node to one `claim` node. Since 87.8% of all nodes
(5,090 / 5,798) have degree exactly 1, a very large share of the graph
consists of these small, low-information attachments, which would dominate
the HAC distance matrix with volume rather than structural signal.

**Issue B — the coherence-circularity hazard (task §T6).** The task requires
that whatever signal is used to measure cluster coherence be independent of
whatever signal drove the clustering. If we cluster using every relation
type and then measure coherence using, say, `cites` or `evaluated_on`
co-membership, that measurement is contaminated, because those same edges
were part of what produced the clusters.

**Decision:** the hyperedge set used to drive clustering excludes two
relation types, which are held out instead:

- **`claims` (690 edges) is excluded from clustering entirely** (Issue A).
  These edges are not deleted from the data; they remain available as a
  potential input to node-level content/context for labeling (T5) but do not
  participate in the structural or semantic similarity computation.
- **`cites` (37 edges) is held out as the independent coherence probe**
  (Issue B). It is small, semantically meaningful (citation is a natural,
  well-precedented independent validation signal in network science), and
  is never used to compute `s_struct`, `s_sem`, or any clustering input. In
  T6, coherence will be measured by checking whether nodes placed in the
  same cluster cite each other (via held-out `cites` edges) more often than
  a degree/arity-preserving null model would predict — this is decided here,
  before implementation, specifically so it cannot be adjusted after seeing
  results.

The remaining 8 relation types (`evaluated_on`, `presents`, `addresses`,
`solves`, `uses_technique`, `authored_by`, `extends`, `uses_component`,
`proposes_future_work`; 653 edges combined at the full 2026 snapshot) form
the hyperedge set actually clustered.

This does not fully resolve the circularity hazard by itself — the semantic
term (`s_sem`) still needs an embedding source independent of whatever the
T5 labeller uses as input, which we will decide when T5 is implemented — but
it commits, at the design stage, to the specific held-out relation for
structural coherence, closing the gap flagged in an earlier draft of this
document.

### Complexity

Let `m = |E'|` be the number of hyperedges **used for clustering** in a
snapshot, i.e. excluding the held-out `claims` and `cites` relation types
(see above). At the full 2026 snapshot, `m ≈ 653` (compared to 1,429 total
hyperedges), smaller than an earlier draft assumed and correspondingly
faster. The distance matrix requires `O(m²)` pairwise similarity
computations; HAC with average linkage is `O(m² log m)` with a standard
priority-queue implementation. For `m ≈ 653` this is on the order of 213,000
pairwise computations, comfortably within the "minutes, not hours" budget
the task specifies. Node assignment is `O(m · d̄)` where `d̄` is the average
node degree (number of incident hyperedges, restricted to the clustered
relation types), negligible by comparison. This method would not scale to
`m` in the hundreds of thousands without approximation (e.g. locality-
sensitive hashing to avoid the full `O(m²)` matrix), but that is explicitly
out of scope per the task's data-size guidance.

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
