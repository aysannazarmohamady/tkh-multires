"""T6 coherence probe: bibliographic coupling between article pairs.
Independent of clustering: `cites` never enters s_struct/s_sem, and no
article's cluster placement ever comes from a `cites` edge (asserted
below: `provenance != "cites"`, not `== "primary"` — articles may be
placed either by real clustering or, deterministically, via the
`presents` method they present, and both are equally untouched by
`cites`). cited_work nodes are used only as reference fingerprints, never
as clustered objects."""
import json, collections, itertools, sys
import numpy as np

def probe(data, hier, level=0, n_perm=10000, seed=0):
    nl = {n["id"]: n for n in data["nodes"]}
    cutoff = hier["cutoff_year"]
    asg, prov = hier["node_assignment"], hier["node_assignment_provenance"]
    refs = collections.defaultdict(set)
    for e in data["hyperedges"]:
        if e["relation_type"] != "cites": continue
        if cutoff is not None and not (e.get("provenance", {}).get("article_year") and
                                        e["provenance"]["article_year"] <= cutoff): continue
        arts = [m for m in e["members"] if m in nl and nl[m]["type"] != "cited_work"]
        cw   = {m for m in e["members"] if m in nl and nl[m]["type"] == "cited_work"}
        for a in arts: refs[a] |= cw
    arts = [a for a in refs if asg.get(a, [None])[level] is not None and len(refs[a]) > 0]
    assert all(prov[a] != "cites" for a in arts), "article assigned via the held-out cites relation -> leak"
    lab = np.array([asg[a][level] for a in arts])
    J = {}
    for i, j in itertools.combinations(range(len(arts)), 2):
        A, B = refs[arts[i]], refs[arts[j]]
        J[(i, j)] = len(A & B) / len(A | B)
    def stat(l):
        w = [v for (i, j), v in J.items() if l[i] == l[j]]
        b = [v for (i, j), v in J.items() if l[i] != l[j]]
        return (np.mean(w) - np.mean(b)) if w and b else np.nan
    obs = stat(lab)
    rng = np.random.default_rng(seed)
    if not np.isfinite(obs):
        # Fix (found by external review): the undefined case (no within-
        # cluster article pairs, e.g. 2020 with only 9 articles) previously
        # fell through to p = 1.0, which looks like a real, high p-value
        # rather than "not computable." Emit None/null explicitly instead.
        return dict(cutoff=cutoff, n_articles=len(arts), n_clusters=len(set(lab)),
                    within_pairs=sum(1 for (i,j) in J if lab[i]==lab[j]),
                    observed=None, null_mean=None, null_sd=None, z=None, p_perm=None)
    null = np.array([stat(rng.permutation(lab)) for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    return dict(cutoff=cutoff, n_articles=len(arts), n_clusters=len(set(lab)),
                within_pairs=sum(1 for (i,j) in J if lab[i]==lab[j]),
                observed=round(float(obs),4), null_mean=round(float(null.mean()),4),
                null_sd=round(float(null.std()),4),
                z=round(float((obs-null.mean())/null.std()),2), p_perm=round(float(p),4))

data = json.load(open("data/tkh_collection10.json"))
for f in sys.argv[1:]:
    print(json.dumps(probe(data, json.load(open(f)))))
