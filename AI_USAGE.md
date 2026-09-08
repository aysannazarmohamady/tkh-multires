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
