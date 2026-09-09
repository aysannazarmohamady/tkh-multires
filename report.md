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

The remaining 9 relation types (`evaluated_on`, `presents`, `addresses`,
`solves`, `uses_technique`, `authored_by`, `extends`, `uses_component`,
`proposes_future_work`; 702 edges combined at the full 2026 snapshot) form
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
(see above). At the full 2026 snapshot, `m = 702` (compared to 1,429 total
hyperedges; an earlier draft estimated `m ≈ 653`, corrected here after
running the actual filter). The distance matrix requires `O(m²)` pairwise
similarity computations; HAC with average linkage is `O(m² log m)` with a
standard priority-queue implementation. For `m = 702` this is on the order
of 246,000 pairwise computations, comfortably within the "minutes, not
hours" budget the task specifies. Node assignment is `O(m · d̄)` where `d̄` is
the average node degree (number of incident hyperedges, restricted to the
clustered relation types), negligible by comparison. This method would not
scale to `m` in the hundreds of thousands without approximation (e.g.
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

### Second review round: findings and fixes

An external review of the implemented method (not just the design) found
four substantive issues, all independently reproduced before fixing:

1. **Circularity leak in `attach_leftover_nodes`.** All 580 `cited_work`
   nodes were assigned via the `cites` relation type — the same relation
   reserved as the independent T6 coherence probe. Testing "do co-clustered
   nodes cite each other more than chance" would have been partly circular
   for exactly these nodes. **Fix:** every node's assignment now carries a
   `provenance` tag (`primary`, `claims`, or `cites`); T6's coherence check
   must filter out `provenance == "cites"` nodes before running. `claims` is
   attempted before `cites` in the attachment order specifically to
   minimize how many nodes end up excluded.
2. **Degenerate level ladder.** An earlier version spaced dendrogram cut
   *heights* linearly between level 0 and the finest level (despite a
   docstring claiming "geometrically" — a real code/docstring mismatch),
   producing cluster counts of 14 → 356 → 660 → 702: essentially one giant
   jump with no usable intermediate level. **Fix:** intermediate levels now
   target cluster *counts* spaced geometrically (e.g. 13 → 49 → 186 → 702),
   each searched for independently, giving an actually usable drill-down
   path.
3. **Level 0 concentration.** With average-linkage, the top 2 of 13-14
   level-0 clusters held 75% of all nodes — technically inside the P2
   count budget but not in the spirit of a useful coarse overview.
   **Fix, evidence-based:** compared average/complete/ward linkage on the
   same distance matrix; complete linkage reduced top-2 concentration to
   47.7% (ward gave a similar 46.6% but was rejected — Ward's method is
   only mathematically valid for Euclidean distances, which our combined
   structural+semantic distance is not). `complete` is now the default
   linkage (`--linkage`, still overridable).
4. **Cross-snapshot embedding reproducibility risk.** An earlier version
   required the embeddings file's edge id set to *exactly* match the
   current snapshot, so only the exact snapshot the Space was run on
   (2026) worked; earlier cutoffs raised `ValueError`, meaning T3 would
   have needed a fresh Space run (and network access) per snapshot. Since a
   hyperedge's embedding depends only on its own members' surface forms,
   not on which snapshot it's viewed from, this was an unnecessary
   restriction. **Fix:** the loader now only requires the current
   snapshot's edges to be a *subset* of the embeddings file's coverage (the
   2026 file, run once, covers all earlier cutoffs since snapshots are
   cumulative), and raises a clear, specific error listing which edges are
   missing if that's ever not the case.

Also corrected: two numeric errors in this document (`m ≈ 653` → `m = 702`;
"8 relation types" → "9") found during the same review.

### Third review round: the coherence probe was undefined, not just leaky

A further review caught that the `provenance`-filtering fix above was
necessary but not sufficient. Every `cites` hyperedge in this dataset has
exactly one `article` member and the rest `cited_work` (confirmed: all 37
edges follow this pattern). Once `cited_work` nodes are excluded as
`cites`-provenance, there is no remaining pair of *filtered-in* nodes that
ever co-occurs in a `cites` edge — the planned "do co-clustered nodes cite
each other more than chance" test has no valid pairs to test on, so it
would not return zero, it would be **undefined**.

**Fix: bibliographic coupling between article pairs**, not co-citation
between arbitrary node pairs. Each article gets a reference fingerprint
(the set of `cited_work` nodes it cites); two articles are compared by
Jaccard overlap of these fingerprints; the statistic is mean within-cluster
Jaccard minus mean between-cluster Jaccard; the null model permutes cluster
labels across articles 10,000 times, preserving cluster sizes and each
article's own reference-set size exactly. `cited_work` nodes are used only
as a fingerprint attribute of an article, never as a clustered object
themselves — implemented as `src/coherence_probe.py`, which asserts every
article's `provenance == "primary"` before running, so a future regression
that re-introduces the leak fails loudly instead of silently.

**Results (permutation test, n=10,000), run on the actual clustering
output:**

| snapshot | articles | within-cluster pairs | observed | null sd | z | p |
|---|---|---|---|---|---|---|
| 2020 | 9 | 0 | — | — | — | undefined (no within-cluster article pairs) |
| 2024 | 25 | 87 | 0.0103 | 0.0068 | 1.52 | 0.076 |
| 2026 | 37 | 126 | -0.0014 | 0.0060 | -0.23 | 0.56 |

**Honest interpretation:** this is a null result — no statistically
significant evidence that co-clustered articles share more references than
chance, at either snapshot where the test is even defined. We report this
directly rather than hiding it: per the task's own framing, "a modest method
honestly and independently evaluated" is preferred over a method whose
numbers can't be trusted. With only 9-37 articles, this probe is
underpowered (only large effects would be detectable) — it is one probe,
not the whole coherence story.

**Correction on a duplicate-count claim from the prior round:** we had
originally cited "122 duplicate `cited_work` entities" from a review by
Claude Opus without independently reproducing it. On direct recount: simple
normalization (`strip().lower()`) finds 0 duplicate surface forms among the
580 `cited_work` nodes; alphanumeric-only normalization finds 16. The
"122" figure, on further explanation from Opus, turned out to count
surface-form *occurrences across multiple `cites` edges* (i.e. it was
measuring the coupling signal itself), not distinct duplicate entities — a
different quantity than what "duplicate" suggested. We use our own directly
verified numbers (0 / 16 depending on normalization strictness) and treat
deduplication as a sensitivity check: merging by alphanumeric normalization
changes the coupling count from 316/666 to 326/666 article pairs sharing a
reference — a small effect, not a pipeline requirement.

**Planned additional probes (not yet implemented — noted here so scope is
explicit), since one 9-37-article probe alone is not enough evidence:**
1. A blind LLM "intruder" test (4 same-cluster members + 1 from a different
   cluster; ask which is the intruder; accuracy vs. the 20% chance rate,
   with confidence intervals). Independent of both the embedding and
   `cites`, and much better powered than bibliographic coupling.
2. A second held-out probe using `authored_by` (38 edges, 250 authors):
   "do co-clustered nodes share an author?" Larger n than the citation
   probe; must be pre-registered (decided now, before seeing results) if
   used, same as `cites` was.
3. Quantify the TF-IDF/embedding correlation (0.307, already measured) as
   an upper bound on how "independent" any embedding-based semantic signal
   really is from structure, to make the "semi-independent" framing
   explicit rather than implicit.

### Note on scope

This document will be extended with the T3 temporal-matching mechanism, the
T4 hyperedge-collapse rule, and full T6 results as those tasks are
completed. This section (Deliverable 0) is considered stable; later
additions will not change the method described above without an explicit
changelog note.
