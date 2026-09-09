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

### T3 — Temporal coupling: mechanism, its real cost, and honest limits

**Why a stabilization mechanism was needed, not just post-hoc matching.**
Two findings made "just match clusters after independently re-clustering
each snapshot" insufficient on its own:
- Cross-snapshot ARI at level 0 (no regularization) was 0.37-0.45 — real
  reshuffling, not the "does not reshuffle every time" behavior the task
  asks for.
- `alpha` sensitivity: ARI between `alpha=0.3` and `alpha=0.7` on the
  *identical* 2026 snapshot was 0.213 — lower than the worst real
  cross-snapshot ARI. An arbitrary hyperparameter choice moves the
  hierarchy more than two years of real corpus growth does. (Also
  confirmed: `alpha=1.0` collapses level 0 to a single cluster,
  `[1, 1, 117, 702]`, an outright P2 violation — empirical evidence for why
  a purely-structural method is unusable given this corpus's 2-3% Jaccard
  density, which is the concrete justification for the P3 resolution.)
- A baseline "pick the level-0 cut height that best matches the previous
  snapshot" (operating only on which height to cut, not the distance
  matrix itself) barely moved cross-snapshot ARI: 0.424 -> 0.449. This
  ruled out "which height to cut" as the source of instability.

**Mechanism chosen: temporal regularization on the hyperedge distance
matrix.** For snapshot `t > 2020`, before running HAC: for every pair of
hyperedges that both existed in snapshot `t-1` and were in the same
level-0 cluster there, reduce their distance by a fixed `lambda`, clipped
to `[0, inf)`. New edges are untouched. `lambda=0` recovers the
unregularized method exactly. Implemented in `src/temporal_reg.py`.

**Effect on real transitions (mean level-0 ARI across the 3 transitions):**

| lambda | mean ARI (real transitions) |
|---|---|
| 0 | 0.424 |
| 0.1 | 0.722 |
| **0.2** | **0.806** |
| 0.4 | 0.815 |

This is a large improvement, and it comes at negligible cost to
within-snapshot fit (intra-cluster distance on the *unregularized* matrix
changes by about 1%, and level-0 cluster count stays within the 10-15
target at every lambda tested) and actually *improves* top-2 concentration
at 2026 (0.534 -> 0.410 at lambda=0.2). Our interpretation: the level-0
dendrogram has several near-tied cut heights, and which one gets picked is
close to arbitrary; regularization resolves this tie-break in favor of
history rather than changing the underlying fit.

**The decisive test, and an honest negative result.** A large ARI gain at
near-zero apparent cost is exactly the profile of a mechanism that might
just be locking in whatever the 2020 clustering happened to be, rather than
genuinely stabilizing against noise. We tested this directly: run the full
2020->2026 chain under 10% random hyperedge removal (5 seeds), compare the
perturbed 2026 output to the unperturbed reference, for both `lambda=0` and
`lambda=0.2`:

| lambda | mean ARI under 10% perturbation (5 seeds) |
|---|---|
| 0.0 | 0.295 ± 0.061 |
| 0.2 | 0.273 ± 0.079 |

**`lambda=0.2` is not better under noise — it is slightly worse than
`lambda=0`** (well within one standard deviation, so not a large effect,
but explicitly not an improvement). This is the real, stated cost of the
mechanism: it is a **tie-break stabilizer for consistent data, not a
noise-robustness mechanism**. It cannot distinguish "this snapshot's
history was arbitrary but not wrong" from "this snapshot's history was
corrupted by the specific edges that got removed" — it reinforces whatever
came before, correct or not. We adopt `lambda=0.2` anyway, because the task
setting is real (not adversarial) corpus growth, where the near-0.81 ARI on
genuine transitions is the more decision-relevant number — but we do not
claim it as a general noise-robustness result, and `outputs/
temporal_events.json` carries this caveat directly rather than only in
prose here.

**Event classification and its reliability ceiling.** Using the
`lambda=0.2` chain's level-0 hyperedge-cluster labels, consecutive
snapshots are matched via hyperedge-set Jaccard overlap (edges present in
both snapshots only), classifying each cluster as a continuation, growth,
merge, split, birth, dissolution, or (new) an explicit "ambiguous weak
link" category for matches that are real but below the confidence
threshold, rather than forcing a classification. Implemented in
`src/build_temporal_events.py`; output in `outputs/temporal_events.json`.
On this corpus, the observed events were births (13, at 2020, the start of
the chain), continuations, growths, and a few ambiguous weak links — no
merges or splits were observed in this run, which we report as-is rather
than searching for a threshold that would produce some. Given the
perturbation-test result above, **~0.28-0.30 ARI is the honest reliability
ceiling for any single event in this log**: some fraction of "continued"
classifications are plausibly arbitrary tie-breaks carried forward by the
regularization rather than real conceptual continuity, and the report does
not claim otherwise.

**Deliberately separated from T1/T2/T6 artifacts.** `outputs/hierarchy_*.json`
(used and verified for T1, T2, and the T6 coherence probe) are produced
with `lambda=0` and are untouched by this section. `temporal_events.json`
uses the separate `lambda=0.2` chain. This keeps the already-verified
non-temporal artifacts stable while still giving T3 a fair mechanism to
evaluate — the two are not silently mixed.

### T2 correction: rank-normalization (the alpha=0.5 scale mismatch)

An external review found that `alpha=0.5` was not actually a balanced
reconciliation between structure and semantics as claimed. We verified
this directly: `s_struct` has mean 0.0044 (zero for 96.7% of pairs) while
`s_sem` has mean 0.300 — very different scales. At `alpha=0.5` on the raw
values, the structural term supplied only 4.7% of the combined
similarity's variance (`var=0.000268` vs `0.005472`, confirmed by direct
computation) — the method was effectively ~95% semantic despite the
`alpha=0.5` framing.

**Fix:** both signals are now rank-normalized to `[0,1]` over their own
upper-triangle *before* mixing (`scipy.stats.rankdata`), so `alpha` controls
the actual mixing weight rather than being dominated by whichever raw
signal has larger variance. **Honest limit of this fix:** re-measuring
after rank-normalization, the structural term's variance share rose to only
8.7%, not 50%. This is not a remaining bug in the normalization — it's a
property of the data: since 96.7% of structural pairs are *exactly* zero,
rank-normalization ties all of them at the same (low) rank, so the
structural signal is inherently low-variance regardless of any monotonic
rescaling. We report this as a genuine, only partially fixable limitation
rather than claiming the rank fix fully balances the two signals.

**Rejected alternative from the same review:** using `authored_by` and
`evaluated_on` as additional "independent" coherence probes for T6. We
checked this against `src/method.py`'s actual relation-type sets and found
both are already among the 9 relation types driving the clustering itself
(`EXCLUDED_FROM_CLUSTERING` only contains `claims`; `HELD_OUT_FOR_COHERENCE`
only contains `cites`) — using either as an "independent" probe would
reintroduce the exact circularity issue already caught and fixed for
`authored_by` in an earlier round. Not implemented.

### T4 — Hyperedge collapse (implementation)

Implemented in `src/hyperedge_collapse.py`, applied to *all* hyperedges
(including `claims` and `cites`, since T4 concerns the graph after
coarsening, independent of which edges fed clustering). For hyperedge `e`
with endpoints landing in super-node set `sigma(e)` (multiplicities `m_S`
= count of e's endpoints in super-node `S`):

- **`|sigma(e)| = 1`:** the edge becomes internal to that super-node. Not
  deleted — recorded in the super-node's `internal_edges`, and counted
  toward a per-cluster cohesion score (internal edge mass / total incident
  mass), usable later for T5/T6.
- **`|sigma(e)| = 2`:** a coarse edge between the two super-nodes, weight
  `1/(|sigma(e)|-1) = 1`, carrying the multiplicity vector `(m_A, m_B)` so
  partial internalization isn't silently lost.
- **`|sigma(e)| >= 3`:** kept as a genuine coarsened **hyperedge** of arity
  `|sigma(e)|` between those super-nodes — not clique-expanded into
  pairwise edges. This is what makes the coarsening hypergraph-native by
  construction rather than a projection. A `clique_expand=True` mode is
  also implemented specifically so the two can be directly compared.

**Verified by hand** (per our established habit): traced hyperedge `h_00050`
(arity 20) through the collapse — 18 of its members land in super-node 6,
1 in super-node 7, 1 in super-node 11, matching the output's
`multiplicities: {"6": 18, "7": 1, "11": 1}` exactly; combined with a
second contributing edge (`h_01365`, also `sigma`-size 3), the aggregated
weight `0.5 + 0.5 = 1.0` matched the output exactly.

**Projection-loss result (level 0, 2026, 14 super-nodes):** 117 native
coarse edges (62 of them true hyperedges of arity >= 3) vs. 83 edges if
clique-expanded — clique expansion collapses 62 genuine multi-way relations
down into far fewer *distinct* pairwise edges (28 new pairs beyond what
native coarsening already had), because with only 14 super-nodes many of
the `C(k,2)` pairs generated from different original hyperedges coincide.
This directly demonstrates the information loss T2 asked us to quantify if
a projection were tried: multiple distinct multi-way relations become
indistinguishable from each other once flattened to pairs.

**What this loses, stated directly:** which specific original members
grounded a coarse relation, and the distinction between e.g. a `(1,1,8)`
endpoint spread and a `(3,3,4)` spread landing in the same 3 super-nodes.
The multiplicity vector is stored specifically so this is at least
recoverable, not silently discarded.

### Note on scope

This document will be extended with the T3 temporal-matching mechanism, the
T4 hyperedge-collapse rule, and full T6 results as those tasks are
completed. This section (Deliverable 0) is considered stable; later
additions will not change the method described above without an explicit
changelog note.
