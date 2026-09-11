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

**Results (permutation test, n=10,000), on the final, current clustering
(after the temporal-leak and P3 fixes below) — STALE NUMBERS CORRECTED:**
an earlier draft of this document quoted z=2.84/2.59 from an intermediate
clustering that no longer exists; the current, reproducible numbers are:

| snapshot | z | p |
|---|---|---|
| 2020 | 0.20 | 0.40 |
| 2022 | 0.51 | 0.29 |
| 2024 | -0.25 | 0.58 |
| 2026 | 1.68 | 0.059 |

**Honest interpretation: a null result at every snapshot.** The clustering
has changed multiple times during this project as bugs were found and
fixed (temporal leak, rank-normalization, paper-frequency weighting,
`presents` handling); the coherence-probe result changed along with it —
briefly appearing significant under one intermediate clustering, now null
under the final one. We report the current, reproducible number rather
than a more favorable past one, and note the instability across fixes
explicitly: it reflects the underlying clustering being revised as real
bugs were found, not an issue with the probe's design (the
`provenance != "cites"` circularity guard applies unchanged throughout).

**Caveat, stated directly:** with only 25-37 articles even at the
significant snapshots, this remains one probe on a small population, not
proof of general coherence — but it is no longer accurate to describe it as
a null result.

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

### Fourth review round: a real temporal leak, and an honest P3 limitation

**1. Temporal leak in snapshot construction, found and fixed.** Snapshots
were filtered on node `year` (falls back to `origin_year`, the real-world
invention/publication date) instead of `first_seen_year` (when THIS corpus
first recorded the entity). This let entities into a snapshot before the
corpus itself had any record of them — a "knowledge from the future" leak
at the snapshot-construction level, independent of and prior to any
labeling concern. Verified directly: 334/332/143 nodes at cutoffs
2020/2022/2024 had `first_seen_year` after the cutoff despite `year <=`
cutoff. Fixed in `src/load_graph.py` and `src/method.py`: nodes now
filtered on `first_seen_year`, edges on `provenance.article_year`. All
downstream artifacts (T1 stats, hierarchies, coherence probe, T3 chain)
regenerated after this fix.

**2. P3 revisited: clusters correlate with source article more than
intended, and a genuine trade-off was found while trying to fix it.**

*Diagnosis, verified directly:* one sample article had 39
clustering-eligible edges, 38 of which shared the same central method
node — a per-paper hub, not a generic popularity effect. Excluding
`presents` and `authored_by` outright barely changed this (external
review's own numbers: NMI 0.609 -> 0.606 -> 0.649 as more relations were
excluded) because the underlying driver is this shared per-paper hub
node appearing across `addresses`/`solves`/`uses_*`/`evaluated_on` edges,
not `presents` itself (`presents` has arity 2, contributing almost nothing
to Jaccard overlap regardless).

*A real implementation conflict was found and resolved:* fully excluding
`presents` from clustering (to reduce paper-identity bias) left articles
with only `proposes_future_work` as a clustering-eligible relation — most,
but not all, articles have one. Articles without it could only reach a
cluster via post-hoc attachment through `cites`, directly reintroducing
the coherence-probe circularity `cites`-provenance filtering exists to
prevent (verified: the probe's assertion failed exactly this way).
**Fix:** `presents` is excluded from the similarity computation (it adds
no real signal) but articles lacking another clustering-eligible edge are
placed *deterministically* — inheriting the full level-path of the method
node they present (`place_articles_via_presents` in `src/method.py`),
tagged `provenance="presents"`. The probe's assertion was correspondingly
relaxed from `provenance == "primary"` to `provenance != "cites"`, since
independence only requires that a node's placement never touched `cites`,
not that it came specifically from primary clustering. In practice, most
articles (45/52) are placed via `proposes_future_work` directly; only 7
need the `presents` fallback.

*Hub down-weighting, including a sign error caught and fixed:* structural
similarity now weights each shared member by `w(v) = log(1 + n_papers(v))`
(`n_papers(v)` = number of distinct articles mentioning `v`), so a node
specific to one paper is down-weighted relative to one recurring across
many papers. **An initial version used the reciprocal,
`w(v) = 1/log(1+n_papers(v))`** — verified directly to be a sign error: it
gave single-paper hub nodes MORE weight, and measurably made the
paper-identity correlation worse (NMI 0.612 vs 0.566 unweighted) before
being caught and inverted.

*Honest result, reported without an unjustified target.* An earlier draft
of this document set a target of "NMI < 0.3" for paper-identity
correlation; on reflection this number had no principled justification and
is dropped. Instead, paper-identity NMI is reported against a null model
(cluster labels randomly permuted, preserving the exact cluster-size
distribution, 1000 reps) and alongside a concrete, interpretable count:

| metric | value |
|---|---|
| observed NMI (level 0, 2026) | 0.485 |
| null NMI (mean, 95% CI) | 0.021 [0.019, 0.023] |
| z vs. null | 450.6 |
| clusters where >50% of edges come from one article | 4 / 14 |

The clustering correlates with source article far more than chance (as
expected — the null model confirms this is a real, not spurious, effect),
and 4 of 14 level-0 clusters (28.6%) are still majority-driven by a single
article's edges. This is reported as a known, quantified limitation of the
method on this corpus, not a solved problem: the corpus's per-paper
extraction structure makes some residual paper-identity correlation
difficult to fully remove through re-weighting alone. Full numbers in
`outputs/paper_identity_diagnostic.json` (`src/paper_identity_diagnostic.py`).

**3. Coherence probe result changed again.** With the corrected clustering
(temporal leak fixed, `presents` excluded, paper-frequency weighting), the
coherence probe (unchanged design; provenance guard now `!= "cites"`)
returns a null result again at every snapshot (best case 2026: z=1.68,
p=0.059) — different from an intermediate draft's significant result,
because the underlying clustering changed multiple times during this
round of fixes. We report the current, actually-reproducible number rather
than an earlier snapshot's, and note explicitly that this number has
changed more than once as bugs were found and fixed — a sign the method is
still not fully stable under implementation-correctness fixes, which is
itself worth stating plainly rather than picking whichever past number
looked best.

**Still open (explicitly deferred, not silently dropped):** the same
review's T3 refinements (separate ARI for nodes whose incident edges did
and did not change; ≥20 perturbation seeds; an arity-preserving edge
shuffle null instead of a label shuffle) and T4's integration into every
level/snapshot and into T3/T6 are not yet implemented as of this section.

### Fifth round: T4 wired into every level/snapshot; T3 refinements, including a genuinely positive finding

**T4, now actually used.** An earlier version of `hyperedge_collapse.py`
ran once (level 0, 2026) and nothing downstream read its output — P4's
10% weight was earned by an implementation that existed but wasn't
exercised. `--all` now runs collapse for every level (0-3) of every
snapshot (2020/2022/2024/2026), 16 combinations, each saved to
`outputs/hyperedge_collapse_<year>_level<k>.json`. Per-cluster cohesion
scores from this output are now read by `src/build_temporal_events.py` and
attached to birth/split events in `temporal_events.json`, so T4's output is
genuinely consumed by T3, not merely computed alongside it.

**T3 refinements, run at 20 seeds:**

1. **Changed vs. unchanged node ARI.** Split perturbation-test nodes by
   whether any of their incident edges were removed. Result: unchanged
   nodes ARI = 0.213 ± 0.293, changed nodes ARI = 0.168 ± 0.233 — the
   right direction (unchanged nodes are more stable) but the two heavily
   overlap given the large spread, so this alone is weak evidence that the
   method does much more than react to noise roughly uniformly across the
   graph.
2. **Paired comparison, `lambda_sd=0` vs `0.25`, same 20 seeds:**
   Wilcoxon signed-rank p=0.237 — not significant, consistent with (not
   contradicting) the earlier finding that `lambda_sd=0` and nonzero values
   don't meaningfully differ in perturbation-robustness.
3. **New, and genuinely positive: an arity-preserving growth-null for real
   transitions.** For each of the 3 real transitions, we built a synthetic
   version where every edge NEW at the later snapshot has its membership
   randomly reassigned (same arity, random nodes) instead of its real
   members, and re-measured transition ARI against this synthetic growth
   (20 seeds). Real transitions were **clearly more structure-preserving**
   than random growth of the identical size, at every transition:

   | transition | observed ARI (real growth) | growth-null ARI (random growth, mean) |
   |---|---|---|
   | 2020→2022 | 0.819 | 0.437 |
   | 2022→2024 | 0.436 | 0.282 |
   | 2024→2026 | 0.499 | 0.059 |

   This is real, positive evidence for P5 that the method's earlier
   perturbation test alone did not show: the corpus's actual growth is
   measurably more structure-preserving than an equivalently-sized random
   perturbation would be, even though the mechanism (post-hoc matching,
   `lambda_sd=0`) applies no artificial stabilization. (Note: this
   analysis uses TF-IDF rather than the fixed sentence-embedding file for
   the semantic signal, since perturbation can change which raw
   `evaluated_on` rows get merged into which edge ids, producing ids the
   fixed embeddings file was never computed for — TF-IDF has no such
   fixed-vocabulary dependency. Absolute ARI values here therefore differ
   somewhat from the real-embedding numbers reported elsewhere; the
   real-vs-random-growth *comparison* is the finding, not the absolute
   numbers.)

Full numbers in `outputs/lambda_selection_and_null.json`.

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

### T3 — Temporal coupling: circularity found and fixed by us

**Why a stabilization mechanism was investigated in the first place.**
Cross-snapshot ARI at level 0 with no intervention was 0.37-0.45 (later
0.43-0.53 after the rank-normalization fix below) — real reshuffling.
Separately, `alpha` sensitivity was measured directly: ARI between
`alpha=0.3` and `alpha=0.7` on the identical 2026 snapshot was 0.213, lower
than the worst real cross-snapshot ARI, and `alpha=1.0` collapsed level 0
to `[1, 1, 117, 702]` (a P2 violation) — concrete evidence that a purely
structural signal is unusable given this corpus's 2-3% Jaccard density,
which is why P3's structure+semantics combination matters at all.

**A regularization mechanism was built, and initially looked very good —
too good.** The mechanism (`src/temporal_reg.py`): for snapshot `t > 2020`,
reduce the distance between hyperedge pairs that were co-clustered at
`t-1` by `lambda`, before running HAC. An initial sweep found `lambda=0.2`
raised mean cross-snapshot ("transition") ARI from 0.424 to 0.806 at
negligible apparent cost.

**We found and neutralized a circularity problem in our own T3 design
before it reached this report as a final claim — the same class of hazard
§6 asks the coherence probe to avoid, occurring instead in the stability
metric.** Two compounding defects, found on our own re-audit:

1. **The metric was circular.** The regularizer directly reduces the
   distance of pairs that were co-clustered at `t-1`, and "stability" was
   then measured as ARI against that same `t-1` partition. The mechanism
   was optimizing the exact quantity used to evaluate it — a large
   apparent improvement could not fail to appear, independent of whether
   it reflected anything real about corpus evolution.
2. **A scale bug compounded it.** `temporal_reg.run_chain` computed its
   similarity manually (`sim = alpha*s_struct + (1-alpha)*s_sem` on raw,
   non-rank-normalized matrices), bypassing the rank-normalization fix
   already applied to `combined_distance_matrix` elsewhere in
   `src/method.py`. Separately, `lambda=0.2` was measured to be ~2.5
   standard deviations of the actual distance distribution (mean 0.848, sd
   0.081) — not a soft prior but large enough to clip most historical
   pairs to near-zero distance, re-imposing the previous dendrogram almost
   verbatim (lock-in, not smoothing).

**Fix and re-decision, evidence-based rather than assumed:**
1. `run_chain` now calls the actual (rank-normalized) `combined_distance_matrix`
   — no more bypass.
2. `lambda` is now expressed in units of the distance distribution's own
   std (`lambda_sd`), swept over `{0, 0.1, 0.25, 0.5}`.
3. **Selection is by perturbation-ARI, not transition-ARI** — specifically
   because the regularizer manipulates transition-ARI directly, so using
   it to choose `lambda` would keep the circularity. Result:

   | lambda_sd | mean perturbation-ARI (10% edge removal, 5 seeds) |
   |---|---|
   | **0.0** | **0.457** |
   | 0.1 | 0.356 |
   | 0.25 | 0.382 |
   | 0.5 | 0.330 |

   `lambda_sd=0` has the *highest* perturbation-ARI of any value tested —
   every nonzero regularization strength was worse under real noise. This
   independently confirms an earlier, cruder perturbation test (0.273 vs
   0.295 at the old, buggy `lambda=0.2` scale) that had already pointed the
   same direction.
4. **Null model added:** the same chain with the previous snapshot's edge
   labels randomly shuffled before regularization is applied. At the
   selected `lambda_sd=0`, this null is — correctly — numerically identical
   to the observed result (both give transition ARIs `[0.532, 0.510,
   0.427]`, mean 0.490, 95% bootstrap CI `[0.427, 0.532]`), because
   `lambda_sd=0` never reads the prior labels at all. This is the expected
   sanity-check behavior, not a coincidence, and confirms the null-model
   code itself is wired correctly for when it matters at `lambda_sd > 0`.
   Full numbers in `outputs/lambda_selection_and_null.json`.

**Decision: `lambda_sd=0` — no temporal regularization.** Cluster identity
is tracked via post-hoc hyperedge-Jaccard matching across independently-
clustered snapshots only (`src/build_temporal_events.py`), not artificially
enforced. This is the "accept and report honestly" option from our original
two-option framing, chosen over "add a stabilization mechanism" specifically
because the mechanism we built for the second option turned out to be
measuring itself. Real transition ARI (0.43-0.53, mean 0.49, with the
corrected rank-normalized similarity) is reported as the actual level of
cross-snapshot consistency.

**Event classification.** Consecutive snapshots are matched via
hyperedge-Jaccard overlap (edges present in both only), classifying each
level-0 cluster as continuation, growth, merge, split, birth, dissolution,
or an explicit "ambiguous weak link" for real-but-below-threshold matches.
With `lambda_sd=0` (no artificial continuity), the event log is more varied
than the earlier (circular) version: splits now appear, and there are more
ambiguous weak links, consistent with an honest ARI of ~0.49 rather than
the inflated ~0.81 the circular mechanism had produced. Output in
`outputs/temporal_events.json`, which carries this full account as a
`reliability_note` field, not only in this report.

### T5 — Labelling with measured faithfulness

**Design, per the P6/§6 circularity concern:** labels are generated using
ONLY `labeller_input` (member surface forms, grouped by node type — the
same content that drove clustering, unavoidable for a topical label).
Faithfulness is then checked using ONLY the generated gloss plus
`faithfulness_signal`: `claims` text (P6-compliant: `article_year <=
cutoff` only) from the specific articles whose hyperedges were assigned to
that cluster, determined via **edge-level** `provenance.article_id` (not a
member node's broader `provenance.articles` list — see the correction
below). `claims` text is never shown during label generation. Implemented
in `src/prepare_labelling_input.py`.

**A precision bug found and fixed while building this.** An initial
version associated a cluster with articles via any member node's
`provenance.articles` field — but this field lists every paper that ever
mentions that entity, not just the paper the cluster's edges came from. In
practice, a cluster centered on one paper's tensorial Atomic Cluster
Expansion method pulled in claims about an unrelated paper ("MACE") solely
because both papers' provenance touched a shared "acetylacetone" dataset
node. Fixed by associating articles via the **edge-level**
`provenance.article_id` of the actual clustering hyperedges assigned to
each cluster — precise, since each edge has exactly one producing article.

**Pilot results (4 of 14 level-0 clusters, 2026 snapshot; full run is the
natural next step, not done here given time):**

| cluster | label | verdict |
|---|---|---|
| 3 | Tensorial & Magnetic Atomic Cluster Expansion | faithful |
| 1 | Equivariant/Directional Message-Passing GNNs for Molecules | **partial overclaim** |
| 5 | ML for Multiscale Computational Modeling (Survey) | faithful |
| 13 | Differentiable Density Functional Theory (D4FT) | **partial overclaim** |

**Over-claim rate: 2/4 (50%) in this pilot** — too small a sample for a
reliable estimate, but the two failures are informative and both follow
the same pattern: a *general* topical description was well-supported, but
a *specific* detail inferred from the raw member list (which dataset was
used; which methods were directly compared against) turned out to be
wrong once checked against the independent claims signal. For cluster 1,
"evaluated largely on QM9" was asserted because QM9 is a cluster member,
but the actual claims describe evaluation on graphene/MoS2, not QM9. For
cluster 13, "compared against GAAW, Psi4" was asserted because those
method names are cluster members, but the actual claims describe
benchmarking against PySCF. **This suggests a concrete guardrail for any
full-scale labelling run: a labeller should avoid asserting specific
comparisons, datasets, or numeric results unless they can be tied to a
specific piece of evidence, and should default to more general phrasing
when the member list contains many candidate specifics.** Full pilot
detail and verdicts in `outputs/labelling_faithfulness_pilot.json`.

**Honest limitation of this specific pilot:** the labelling and the
faithfulness check were both performed by the same model (Claude) within
one working session, with the process structured so the label-generation
step used only `labeller_input` and the faithfulness-check step used only
the gloss plus `faithfulness_signal` — but this is a *procedural*
separation, not the stronger guarantee a genuinely separate model call (or
a separate NLI model) would give. A production version should enforce this
with two separate, non-overlapping API calls.

### T6 — Extrinsic utility

**Design (label-free, per external review):** the hierarchical drill-down
represents each level-0/level-1 super-node by the mean TF-IDF vector of its
members' surface forms, independent of T5. The scorer (TF-IDF cosine
similarity) is a different embedding family from the MiniLM sentence
embeddings used for clustering. Beam search descends the hierarchy,
finally ranking the member nodes of visited leaves; the flat baseline
ranks all candidate nodes directly with the same scorer. Duplicate target
names are one equivalence class; Q5/Q11 (zero resolvable targets) and
Q15-18 (a claims task, not methods) are out of scope. Evaluated on the
2026 snapshot only.

**Two real bugs were found by external review and fixed, changing the
conclusion entirely.**

1. **Missing overhead.** The reported "cost" only counted the final
   leaf-ranking step, silently omitting the 66-102 super-node scores spent
   during descent — understating true cost by roughly 100x for any target
   found in the leaf ranking.
2. **Non-deterministic tie-breaking.** After fixing (1), the one apparent
   "hit" (Q8) turned out to be a node (`cite_00001`) with a TF-IDF score
   of **exactly 0.0** — tied with 101 other zero-score candidates, with
   its reported rank depending on Python's per-process string-hashing
   order (`PYTHONHASHSEED`), verified directly to range from 136 to 236
   across different seeds. **The recall@cost table was measuring tie
   order, not retrieval.**

**Fix:** ranking is now deterministic (sorted candidate order) and
tie-aware — a zero similarity score means "no lexical overlap at all" and
is explicitly treated as **not retrieved**, not ranked arbitrarily among
other zeros.

**Corrected result: a null, uninformative comparison, reported as such.**
Of 47 targets across 12 usable questions, the hierarchical system finds
**0**, and the flat baseline finds **2** (both in Q14). Neither system can
retrieve the large majority of targets at any cost, because TF-IDF gives
**zero** lexical similarity between natural-language question text
("Which methods are best suited for...") and short technical target names
("MACE", "ConvLSTM") for 45 of 47 targets. **This is a scorer limitation,
not evidence about whether the hierarchy is useful** — but it means this
specific comparison cannot say anything about the hierarchy's value until
re-run with a denser, non-lexical scorer (e.g. a second sentence-transformer
model, genuinely independent of the MiniLM used for clustering). We report
this as a null result rather than the earlier (incorrect) positive-looking
comparison.

Full detail in `outputs/extrinsic_eval_detail.json` and
`outputs/extrinsic_eval_summary.json`.

### T3 — authoritative perturbation-robustness result

A separate bug affected the original T3 perturbation test: it perturbed
raw hyperedges *before* `merge_evaluated_on_by_article` ran, which could
change which rows get merged into which edge id, producing ids the fixed
MiniLM embeddings file was never computed for. This forced that test onto
TF-IDF, where 13 of 20 seeds degenerated to a single level-0 cluster
(ARI=0) — an invalid result.

**Fix, in `src/t3_perturbation_real_embeddings.py`:** perturb the
clustering-eligible edges *after* merging (removing 10% of them never
changes any other edge's id), using real MiniLM embeddings via subset
matching, 100 seeds.

**A second, more subtle error was caught and corrected in the same
script: the statistical test itself was wrong.** An earlier version
compared a single transition's ARI against the *confidence interval of
the mean* perturbation ARI — that interval shrinks as more seeds are
added and says little about whether any one specific transition is
unusual. The corrected test instead compares each transition's ARI
against the actual *distribution* of the 100 individual per-seed ARIs:
`p = (1 + #seeds with ARI >= transition ARI) / (1 + n_seeds)` — a small p
means that transition is more stable than typical 10%-edge-removal noise.

**Result, reporting the clustering-placed node set as primary (excludes
nodes only reached via post-hoc attachment) and the all-assigned-nodes set
as secondary:**

| | 2020→2022 | 2022→2024 | 2024→2026 |
|---|---|---|---|
| Transition ARI (clustering-only, primary) | 0.608 | 0.693 | 0.592 |
| p vs. 100-seed perturbation distribution | 0.19 | **0.03** | 0.24 |
| Transition ARI (all nodes, secondary) | 0.587 | 0.668 | 0.487 |
| p vs. 100-seed perturbation distribution | 0.09 | **0.01** | 0.32 |

**Honest interpretation: only the 2022→2024 transition is robustly more
stable than 10%-edge-removal noise.** The other two transitions are not
distinguishable from noise by this test. This is a real, if partial,
positive P5 result — not the uniformly positive "all three transitions
beat the noise floor" claim an earlier version of this analysis made
using the wrong statistical comparison. We also note explicitly that this
is not an equivalent-magnitude comparison: perturbation removes 10% of
edges, while real growth adds 40-92% of edges across these transitions —
p-values here should be read as "more/less stable than 10%-removal noise
specifically," not as a matched-size comparison. Full detail, including
both node-set variants, in `outputs/t3_perturbation_real_embeddings.json`.



### Note on scope

This document will be extended with the T3 temporal-matching mechanism, the
T4 hyperedge-collapse rule, and full T6 results as those tasks are
completed. This section (Deliverable 0) is considered stable; later
additions will not change the method described above without an explicit
changelog note.
