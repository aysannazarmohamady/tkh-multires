# AI Usage Log

This file tracks where AI assistants were used, the prompts that drove the
work, what was accepted or modified, and what was verified. Updated
incrementally as work proceeds.

## T1 - Load and describe the evolving graph

**Tool:** Claude - Sonnet 5 

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
