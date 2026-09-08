# AI Usage Log

This file tracks where AI assistants were used, the prompts that drove the
work, what was accepted or modified, and what was verified. Updated
incrementally as work proceeds.

## T1 - Load and describe the evolving graph

**Tool:** Claude - sonnet 5 Medium

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
