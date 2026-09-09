# AI Usage Log

This file tracks where AI assistants were used, the prompts that drove the
work, what was accepted or modified, and what was verified. Updated
incrementally as work proceeds.

## T1 - Load and describe the evolving graph

**Tool:** Claude (Anthropic)

**Major prompts (paraphrased from the actual session):**
1. "Write a script that loads the TKH graph and describes it (T1): node/edge
   counts, node-type distribution, hyper-edge arity distribution, and growth
   between snapshots, sliced into at least three time cutoffs."
2. After I ran it and manually verified the results: "Add a check for the
   edge/node year inconsistency I found (137 hyper-edges where the edge year
   is earlier than the max year among its member nodes)."

**What it helped with:**
- Wrote the initial version of `src/load_graph.py`: cumulative snapshot
  construction by year cutoff, per-snapshot descriptive stats, and growth
  stats between snapshots.
- Proposed the snapshot-inclusion rule (an edge only counts once all of its
  member nodes are present in that snapshot), which is what surfaces the
  `dropped_partial_edges` count.
- Added the edge/node year inconsistency check after I identified the issue
  myself and asked for it to be automated.
- Rewrote the module docstring, after my feedback, to explain the reasoning
  behind the snapshot rule and the alternative that was rejected, instead of
  summarizing the code.

**What I verified myself:**
- Manually traced 3 of the 31 edges dropped at the `cutoff=2020` snapshot in
  the raw JSON and confirmed each one referenced a member node whose year was
  genuinely later than the cutoff.
- Cross-checked the final `cutoff=2026` totals (5798 nodes, 1429 hyper-edges)
  against a direct count of the raw JSON lists; they matched exactly.
- Found the edge/node year inconsistency myself while doing the manual check
  above, then confirmed the automated count (137/1429) matched what I found
  by hand before accepting the added check.

## T2 prep — Literature review (paper search)

**Tools:** Claude (Anthropic), for both search and drafting; a second AI
assistant (ChatGPT), used independently for a cross-check.

**Process:**
- Used AI-assisted web search to find candidate papers matching the four
  required areas (hierarchical community detection, hypergraph
  clustering/spectral partitioning, hypergraph coarsening, dynamic community
  detection), combined with my own manual judgment on which candidates were
  topically relevant enough to pursue.
- Asked a second AI assistant to independently re-verify the candidate list
  against primary sources (existence, correct venue/year, actual content vs.
  claimed content), rather than trusting the first pass at face value.
- Made the final inclusion/exclusion decisions myself based on both AI
  outputs plus my own direct verification (see below), not by accepting
  either AI's conclusion automatically.

**What I verified myself:**
- The second AI assistant's cross-check flagged one paper (TSA-HGNN,
  Vusirikkayala & Viswanatham, Frontiers in AI 2026) as a likely
  fabrication, since a quick search had not surfaced it. Rather than
  removing it on that basis alone, I fetched the paper's PubMed record
  directly (PMID 42290696, PMCID PMC13261176, DOI 10.3389/frai.2026.1824901)
  and confirmed it is a real, indexed publication. The false alarm was most
  likely due to the paper being very recent (May 2026) and not yet indexed
  everywhere. This is recorded in `literature_review.md` under
  "Verification note."
- Independently confirmed HySpecPro's venue: the arXiv listing notes it was
  accepted to DAC 2026, which is not one of the venues the task names as
  preferred (KDD/NeurIPS/ICLR/WWW/TKDE), and I flagged this in
  `literature_review.md` rather than overstating the venue.
- Accepted the correction that HyperSF (2021) falls outside the task's
  suggested 2022-2026 window, and wrote an explicit justification in
  `literature_review.md` for keeping it anyway (best topical fit for T4),
  rather than silently including an out-of-window paper.

## T2 — Method design and Deliverable 0

**Tool:** Claude (Anthropic)

**Role:** In this part of the work, Claude acted as a research assistant:
searching for and surfacing candidate algorithms and papers on request. I
directed the process and made the actual decisions about what to keep,
reject, or push further on.

**Major prompts (paraphrased):**
1. "Search the world of algorithms for an idea to complete and optimize a
   combined structural+semantic clustering approach."
2. "Are you sure there isn't a better idea for this?" — asked repeatedly,
   after each candidate Claude surfaced (Leiden-based multi-resolution
   clustering, then hypergraph modularity/h-Louvain, then hyperedge-based
   hierarchical clustering), because none of the first attempts were good
   enough on inspection.
3. "Write Deliverable 0 based on this literature."

**What Claude helped with:**
- Searched for and surfaced candidate algorithms and papers at each round:
  Leiden/Louvain multi-resolution clustering, hypergraph modularity
  (h-Louvain), and hyperedge-based hierarchical agglomerative clustering
  (Lotito et al. 2023; DeWolfe & Théberge 2025).
- Fetched full papers (not just search snippets) and reported their actual
  content, including limitations, when asked to confirm a candidate was
  solid.
- Drafted the formal Deliverable 0 statement (objective, complexity, the
  guarantees-vs-empirical table, and the rejected-alternative justification)
  once a method was selected.

**What I decided:**
- I rejected the first candidate (Leiden) because it required projecting the
  hypergraph to a pairwise graph, which conflicts with P4.
- I rejected the second candidate (hypergraph modularity/h-Louvain) after
  Claude reported that even the method's own authors state it "often fails
  to find meaningful communities" without extra tuning, which made it too
  heavy to implement soundly in this task's time budget.
- I selected the third candidate (hyperedge-based HAC) as the final basis
  for the method, and directed that its two known gaps relative to our task
  (overlapping communities conflicting with P1, and no semantic signal)
  be addressed explicitly in the design rather than glossed over.

**Still outstanding:** the formal guarantees in Deliverable 0 (e.g. P1
laminarity, P2's achievable cluster count, P5 stability) are stated based on
known properties of hierarchical agglomerative clustering, not yet confirmed
by running the actual implementation on our data. That confirmation is the
next step before treating any of these guarantees as established for this
project specifically.

## T2 — Self-review of Deliverable 0 and fixes

**Context:** After drafting Deliverable 0, I conducted my own review of the
design before allowing any implementation to proceed, checking the proposed
similarity function and node-assignment rule against the actual dataset
rather than accepting them on paper.

**What I found and brought to Claude to verify and fix:**
1. The similarity formula `sim = s_struct^α · s_sem^(1-α)` (geometric mean)
   is degenerate on this dataset: only ~2% of hyperedge pairs share any
   member node, so `s_struct = 0` for ~98% of pairs, which zeroes out the
   entire similarity for any `α > 0`.
2. 87.8% of nodes have degree 1, making the "largest share of incident
   hyperedges" node-assignment rule trivial (vacuous) for most of the graph.
3. The coherence-circularity hazard (task §T6, the highest-weighted
   criterion) was left unaddressed in the frozen Deliverable 0.
4. The provided `questions.csv` / `ground_truth.json` extrinsic-evaluation
   data had not been copied into the repo or used at all.
5. `requirements.txt` needed to be confirmed present in the actual repo.

**What Claude did:**
- Independently recomputed each of the above directly against
  `data/tkh_collection10.json` to confirm the figures (2.03% nonzero
  Jaccard pairs; 87.8% degree-1 nodes; 277 duplicate surface forms found
  as a related issue) before accepting any of them as real.
- Rewrote the Deliverable 0 similarity function to an arithmetic-mean
  combination (fixing point 1), added an explicit note on why the
  node-assignment rule is trivial for degree-1 nodes rather than treating
  it as a hidden flaw (point 2), and designed a concrete fix for point 3
  (holding out the `cites` relation type as an independent structural
  signal for T6 coherence measurement, and excluding the near-degenerate
  `claims` relation type from clustering entirely).
- Extended `src/load_graph.py` to report degree distribution, singleton
  fraction, duplicate surface forms, and growth-by-type, so these
  properties are visible from T1 output going forward rather than only
  discovered by manual audit.

**What I verified myself after the fix:** re-ran the updated `load_graph.py`
in Colab and confirmed the reported `singleton_fraction` (0.878) and the
duplicate surface form example (`"graph neural networks"`, 4 ids) matched
what I had found in my own review, before accepting the fix as correct.

## T2 — Real semantic embeddings via a Hugging Face Space

**Context:** After implementing the method with the offline TF-IDF fallback,
I checked the correlation between the structural and semantic similarity
signals and found it was 0.691 — high enough to undermine the claim that
combining them adds real independent information (see the P3 discussion in
`report.md`). I have my own Hugging Face account, so I decided to compute
real sentence-transformer embeddings there instead of relying on the offline
TF-IDF fallback.

**What Claude helped with:**
- Wrote `compute_embeddings_hf.py` (a standalone script) and
  `hf_space/app.py` (a Gradio app for the same computation, deployed as a
  Hugging Face Space), both using the exact same hyperedge filtering as
  `src/method.py` (excludes `claims`, holds out `cites`) so the embeddings
  line up with what the main pipeline expects.
- Debugged a ZeroGPU startup error on the Space ("No @spaces.GPU function
  detected") by adding a guarded `@spaces.GPU` decorator that no-ops outside
  a ZeroGPU environment.
- Extended `src/method.py` with `semantic_similarity_from_embeddings()` and
  `--embeddings-path` / `--embeddings-order` CLI options, so it can consume
  the Space's output instead of TF-IDF, with an explicit error if the edge
  id sets don't match (rather than silently misaligning rows).

**What I verified myself:** after running the Space on
`data/tkh_collection10.json` (cutoff 2026) and downloading its two output
files, I had Claude recompute the structural/semantic correlation with the
real embeddings in place of TF-IDF. It dropped from 0.691 to 0.307,
confirming the real embeddings are a meaningfully more independent signal
than TF-IDF, which is why the repo uses them (via the Space) as the
preferred path, with TF-IDF kept only as an offline fallback for
environments without Hugging Face access.

## T2 — Second external review of the implemented method

**Context:** After committing `src/method.py`, I had it reviewed again
(independently of my earlier design self-review) — this time checking the
actual running code and outputs, not just the design in `report.md`.

**What was found and what I directed Claude to verify and fix:**
1. `attach_leftover_nodes()` assigned all 580 `cited_work` nodes via the
   `cites` relation — the same relation reserved as T6's independent
   coherence probe, reopening the circularity issue we thought we'd closed.
2. The level-extraction docstring claimed "geometric" spacing but the code
   used `np.linspace` (linear), producing a degenerate cluster-count ladder
   (14 → 356 → 660 → 702) with no usable intermediate level.
3. Level 0's top 2 clusters (of 13-14) held 75% of all nodes — inside the
   P2 count budget but not a useful coarse overview in practice.
4. Loading precomputed embeddings required an exact edge-id-set match, so
   only the exact snapshot the HF Space was run on worked; earlier
   snapshots raised `ValueError`, which would have forced a fresh Space run
   per snapshot in T3.
5. Two numeric errors in `report.md` (`m ≈ 653` vs actual 702; "8 relation
   types" vs actual 9) and a stale "placeholder" comment in
   `requirements.txt`, plus unused dependencies (`networkx`, `pandas`)
   listed there.

**What Claude did:**
- Independently reproduced every claim above against the real code and data
  before changing anything (e.g. recomputed the 75% concentration and the
  364/1839 unassigned-node count at the 2020 cutoff, both matched exactly).
- Added a `provenance` field per node (`primary` / `claims` / `cites`) so T6
  can filter out `cites`-attached nodes from the coherence probe, with
  `claims` attempted first to minimize how many nodes are excluded.
- Rewrote level extraction to target geometrically-spaced cluster *counts*
  directly (fixing both the docstring mismatch and the degenerate ladder in
  one change).
- Compared average/complete/ward linkage empirically (top-2 concentration:
  75% / 47.7% / 46.6%) and switched the default to `complete`, rejecting
  `ward` despite similar numbers because it isn't mathematically valid for
  a non-Euclidean distance.
- Relaxed the embeddings loader to require the current snapshot's edges to
  be a subset of the embeddings file's coverage, rather than an exact
  match, removing the need to re-run the Space per snapshot.
- Fixed the two numeric errors in `report.md`, rewrote `requirements.txt`
  to list only what `src/` actually imports, pinned to the versions
  actually used, and removed the stale comment.

**What I verified myself:** re-ran `src/method.py` on the 2026 snapshot
after all fixes and confirmed the new cluster-count ladder (13 → 49 → 186 →
702), the reduced top-2 concentration (47.7% with complete linkage), and
that the 2020 snapshot now runs against the 2026 embeddings file without
error, before accepting the fixes as complete.

## T6 — Third review round: coherence probe was undefined, not just leaky

**Context:** After the provenance fix in the second review round, I asked
for a third check specifically on whether the planned T6 coherence probe
("do co-clustered nodes cite each other more than chance") was actually
computable once `cites`-provenance nodes were filtered out.

**What was found:** every `cites` edge has exactly one `article` and the
rest `cited_work` members (verified: all 37 edges). Once `cited_work` is
excluded (its provenance is always `cites`), no pair of remaining nodes
ever co-occurs in a `cites` edge — the probe as originally planned would be
undefined, not merely zero.

**The fix (bibliographic coupling) and its script (`coherence_probe.py`)
were written by Claude Opus** (a separate model/session from the one doing
this implementation work), given the same problem description (the
undefined-probe issue above): compare articles' reference sets (Jaccard)
instead of raw node co-citation; permutation null (10,000 reps) preserving
cluster sizes and each article's reference-set size; an `assert` that every
article used is `provenance == "primary"`, so a future regression fails
loudly instead of silently.

**What I verified myself before accepting it:**
- Confirmed the cites-edge structure claim directly against the data (37/37
  edges match the 1-article-plus-cited_work pattern).
- Ran the provided script against our own `outputs/hierarchy_2020.json`,
  `hierarchy_2024.json`, and `hierarchy.json` (2026) and reproduced the
  externally-reported table exactly (2026: z=-0.23, p=0.56; 2024: z=1.52,
  p=0.076; 2020: undefined, 0 within-cluster article pairs) before writing
  any of it into `report.md`.
- Separately checked a number claimed by Claude Opus during its review
  ("122 duplicate `cited_work` entities") and could not reproduce it under
  several reasonable normalizations (0 with simple normalization, 16 with
  alphanumeric-only). Followed up and learned the "122" was actually
  counting something else (surface-form occurrences across multiple `cites`
  edges — the coupling signal itself, not duplicates). Used my own directly
  verified numbers instead of the unverified claim.
- Confirmed the dead `target_mid` variable in
  `find_threshold_for_cluster_range` was unused and removed it.
- Re-ran `src/method.py` on 2020, 2022, and 2024 (in addition to 2026) to
  get real per-snapshot unassigned-node counts and level-0 concentration
  numbers for the README, rather than only reporting the 2026 figures.

**What I accepted as-is:** the coherence probe's statistical design
(Jaccard-based bibliographic coupling, the specific permutation null) was
used as provided, since it is a standard bibliometric technique and its
logic (compare within- vs between-cluster average Jaccard, permute cluster
labels for the null) was straightforward to check by reading the script.

**Honest result:** the probe returns a null result (no significant
coherence signal detected at 2024 or 2026, undefined at 2020). This is
reported as-is in `report.md` rather than adjusted or hidden, along with an
explicit note that a 9-37-article probe is underpowered and should not be
the only coherence evidence — three additional, cheaper probes are planned
(blind LLM intruder test, `authored_by` co-membership, and quantifying the
TF-IDF/embedding correlation as an independence upper bound) but not yet
implemented.
