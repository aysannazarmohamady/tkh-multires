"""
Static figures for report.md (optional deliverable §6.6; no interactive UI).

  outputs/fig_t3_stability.png  -- per-snapshot 10%-removal null ARI
      distributions with the observed cross-snapshot transition ARIs.
  outputs/fig_level0_evolution.png -- level-0 super-node sizes per snapshot,
      coloured by persistent id (identity tracked by temporal_events.json).

Usage:
    python src/make_figures.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CUTOFFS = [2020, 2022, 2024, 2026]
PATHS = {2020: "outputs/hierarchy_2020.json", 2022: "outputs/hierarchy_2022.json",
         2024: "outputs/hierarchy_2024.json", 2026: "outputs/hierarchy.json"}


def fig_t3():
    t3 = json.load(open("outputs/t3_perturbation_real_embeddings.json"))
    nulls = [t3["null_by_snapshot"][str(c)]["aris_clustering_only"] for c in CUTOFFS]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.boxplot(nulls, positions=range(len(CUTOFFS)), widths=0.5, showfliers=False)
    for i, r in enumerate(t3["transitions_clustering_only_PRIMARY"]):
        ax.plot([i, i + 1], [r["observed_ari"]] * 2, "o-", color="C3", lw=2,
                label="observed transition ARI" if i == 0 else None)
        ax.annotate(f"p_Holm={r['p_conservative_holm']:.2f}", (i + 0.5, r["observed_ari"] + 0.02),
                    ha="center", fontsize=8, color="C3")
    ax.set_xticks(range(len(CUTOFFS)), [str(c) for c in CUTOFFS])
    ax.set_ylabel("Level-0 ARI (clustering-placed nodes)")
    ax.set_title(f"10%-edge-removal null per snapshot ({t3['n_seeds']} seeds) vs real transitions", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("outputs/fig_t3_stability.png", dpi=150)


def fig_level0():
    per_snap = {c: sorted((s for s in json.load(open(PATHS[c]))["super_nodes"] if s["level"] == 0),
                          key=lambda s: int(s["persistent_id"][1:])) for c in CUTOFFS}
    pids = sorted({s["persistent_id"] for sns in per_snap.values() for s in sns}, key=lambda p: int(p[1:]))
    cmap = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    colour = {pid: cmap[i % len(cmap)] for i, pid in enumerate(pids)}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for x, c in enumerate(CUTOFFS):
        bottom = 0
        for s in per_snap[c]:
            size = len(s["member_ids"])
            ax.bar(x, size, bottom=bottom, color=colour[s["persistent_id"]], edgecolor="white", width=0.6)
            if size >= 150:
                ax.text(x, bottom + size / 2, s["persistent_id"], ha="center", va="center", fontsize=6)
            bottom += size
    ax.set_xticks(range(len(CUTOFFS)), [str(c) for c in CUTOFFS])
    ax.set_ylabel("assigned nodes (stacked by level-0 super-node)")
    ax.set_title("Level-0 super-nodes by persistent id (same colour = same tracked identity)", fontsize=9)
    fig.tight_layout()
    fig.savefig("outputs/fig_level0_evolution.png", dpi=150)


if __name__ == "__main__":
    fig_t3()
    fig_level0()
    print("Saved outputs/fig_t3_stability.png, outputs/fig_level0_evolution.png")
