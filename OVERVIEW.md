# Project Overview (Plain-Language Summary)

## What is this project about?

We have a large graph of scientific knowledge (from research papers) called a
**Temporal Knowledge Hypergraph (TKH)**. In this graph:

- **Nodes** are things like methods, datasets, tasks, metrics, and papers.
- **Edges** are not simple lines between two nodes — they are **hyper-edges**,
  meaning one edge can connect *several* nodes at once (for example, one edge
  might link a method, a dataset, and a metric together, because a paper
  evaluated that method on that dataset using that metric).
- Every node and edge has a **year** attached, and the graph **grows over time**
  as new papers get added.

Our actual dataset has about **5,800 nodes** and **1,400 hyper-edges**, built
from 52 articles, spanning years from 1901 to 2026 (most of the real activity
is recent).

## What's the problem we're solving?

If you tried to look at all 5,800 nodes at once, it would be overwhelming and
useless — like trying to read a phone book to understand a city. We want to
build a **zoomable summary** of this graph, similar to zoom levels on a map:

- **Zoomed out (top level):** ~10-15 broad, meaningful groups (e.g., "graph
  neural networks", "reinforcement learning methods", etc.)
- **Zoomed in (deeper levels):** more specific sub-groups, down to individual
  methods, datasets, and papers.

This is called a **multi-resolution hierarchy** — a tree-like structure where
each level is a coarser or finer view of the same graph.

## What makes this hard (and interesting)?

1. **The groups must actually make sense to a human expert** — not just be
   "things that happen to be near each other" in the graph. A group should
   represent a real, coherent scientific idea.

2. **We can't oversimplify the hyper-edges.** Many tools convert these
   multi-node edges into a bunch of simple two-node edges to make the math
   easier — but that throws away real information (e.g. "these 3 things were
   evaluated together" becomes 3 separate, weaker facts). We need to work with
   the hyper-edges as they really are.

3. **The graph changes over time**, and every time new papers are added, we
   don't want the whole hierarchy to be rebuilt from scratch and reshuffled.
   Instead, we want most of it to stay stable, and only the parts that
   actually changed (new topics, merged ideas, etc.) to update. We also need
   to track the "life story" of each group over time: when it was born, when
   it grew, when it merged with another group, when it split, or when it
   effectively disappeared.

4. **Every group needs an honest, short description** (like a title + one
   sentence), generated automatically — but it must not claim anything that
   isn't actually supported by the papers/nodes inside that group, and it must
   not "know" anything from after that point in time.

5. **We must evaluate all of this carefully and honestly.** This is the part
   that carries the most weight. In particular, we must avoid a subtle trap:
   if we use one signal (e.g. a specific way of measuring similarity) to build
   the groups, we can't turn around and use that *same* signal to prove the
   groups are good — that would be circular reasoning. We need independent
   checks, comparisons against random/baseline groupings, and a real
   downstream use-case test (e.g., "does this hierarchy actually help someone
   find something faster than browsing the flat graph?").

## What do we need to produce at the end?

- Working code that builds this multi-resolution hierarchy from the data.
- A hierarchy file for each time period (who's grouped with whom, and why).
- A log of how groups changed over time (born / grew / merged / split / died).
- A set of evaluation numbers (with honest, non-circular metrics).
- A short written report explaining our design choices, trade-offs, what we
  are confident about, and what we're still unsure about.

## In one sentence

We're building a trustworthy, zoomable, time-aware summary of a large
scientific knowledge graph — one that respects its true multi-way
relationships and stays stable as new research gets added.
