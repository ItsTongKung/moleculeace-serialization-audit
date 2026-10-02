"""A-6  Compound-level criterion reachability, cliff-pair composition, and the
size mechanism.  Regenerates, from the analysis cache, the descriptive numbers
that Figure 1 rests on.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem

CACHE = os.path.join(C.OUT, "cache")
rows = []
BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]
binlen = {b: [[], []] for b in BINS}     # [admitted lengths, not-admitted lengths]
pool_all = []
for ds in C.datasets():
    df = C.load(ds)
    smi = df['smiles'].tolist()
    n = len(smi)
    az = np.load(os.path.join(CACHE, "adj_" + ds + ".npz"))
    Tb = np.unpackbits(az['packed_Tb'], axis=1)[:, :n].astype(bool)
    Sb = np.unpackbits(az['packed_Sb'], axis=1)[:, :n].astype(bool)
    Lb = np.unpackbits(az['packed_Lb'], axis=1)[:, :n].astype(bool)
    Fb = np.unpackbits(az['packed_Fb'], axis=1)[:, :n].astype(bool)
    Tm = np.load(os.path.join(CACHE, "T_" + ds + ".npy"))
    D0 = C.compound_labels(Tb | Sb | Lb, Fb).astype(bool)
    # compound-level reachability: is this cliff compound reachable by branch X ALONE?
    reach = {'ecfp': C.compound_labels(Tb, Fb).astype(bool),
             'scaf': C.compound_labels(Sb, Fb).astype(bool),
             'str':  C.compound_labels(Lb, Fb).astype(bool)}
    D2 = C.compound_labels(Tb | Sb, Fb).astype(bool)
    r = dict(dataset=ds, n=n, n_cliff_D0=int(D0.sum()))
    for k, v in reach.items():
        r["pct_reach_" + k] = 100.0 * float((v & D0).sum()) / max(int(D0.sum()), 1)
    r['pct_lost_without_string'] = 100.0 * float((D0 & ~D2).sum()) / max(int(D0.sum()), 1)
    r['pct_relabelled_D0_to_D2'] = 100.0 * float((D0 != D2).mean())

    # size mechanism, within matched ECFP4-similarity bins
    iu = np.triu_indices(n, 1)
    slen = np.array([len(s) for s in smi], float)
    plen = ((slen[:, None] + slen[None, :]) / 2.0)[iu]
    ts = Tm[iu]
    ladm = Lb[iu]
    for b in BINS:
        m = (ts >= b[0]) & (ts < b[1])
        if m.sum() > 50:
            binlen[b][0].append(plen[m & ladm])
            binlen[b][1].append(plen[m & ~ladm])
    # heavy-atom quartile cliff rates
    hac = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in smi])
    q = np.quantile(hac, [0, .25, .5, .75, 1.0])
    for i, (lo, hi) in enumerate(zip(q[:-1], q[1:])):
        m = (hac >= lo) & (hac <= hi) if i == 3 else (hac >= lo) & (hac < hi)
        r["D0_cliffrate_Q%d" % (i + 1)] = 100.0 * float(D0[m].mean())
        r["D2_cliffrate_Q%d" % (i + 1)] = 100.0 * float(D2[m].mean())
    # all cliff pairs' ECFP4 similarity, for the below-0.5 count
    cp = Fb[iu] & (Tb | Sb | Lb)[iu]
    pool_all.append(ts[cp])
    rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(C.OUT, "A6_composition.csv"), index=False)
L = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)

W("=" * 104)
W("A-6  COMPOUND-LEVEL CRITERION REACHABILITY AND THE SIZE MECHANISM")
W("=" * 104)
for k, nm in [('ecfp', 'ECFP4 Tanimoto >= 0.9'), ('scaf', 'generic-scaffold >= 0.9'),
              ('str', 'SMILES-Levenshtein >= 0.9')]:
    c = d["pct_reach_" + k]
    W("  cliff compounds reachable by %-28s median %6.2f%% (IQR %.2f-%.2f)"
      % (nm, c.median(), c.quantile(.25), c.quantile(.75)))
W("  cliff labels lost if the string criterion is removed: median %.2f%% (IQR %.2f-%.2f)"
  % (d.pct_lost_without_string.median(), d.pct_lost_without_string.quantile(.25),
     d.pct_lost_without_string.quantile(.75)))
W("  compounds relabelled D0 -> D2: median %.2f%% (range %.2f-%.2f)"
  % (d.pct_relabelled_D0_to_D2.median(), d.pct_relabelled_D0_to_D2.min(),
     d.pct_relabelled_D0_to_D2.max()))
allcp = np.concatenate(pool_all)
W("  cliff pairs with ECFP4 similarity < 0.5, admitted as 'similar at >= 0.9': %d of %d (%.1f%%)"
  % (int((allcp < 0.5).sum()), allcp.size, 100 * float((allcp < 0.5).mean())))
W("")
W("SIZE MECHANISM: mean SMILES length of Levenshtein-ADMITTED vs NOT-admitted pairs, within ECFP4 bins")
sz = []
for b in BINS:
    A = np.concatenate(binlen[b][0]) if binlen[b][0] else np.array([])
    Bn = np.concatenate(binlen[b][1]) if binlen[b][1] else np.array([])
    if A.size > 30 and Bn.size > 30:
        W("  ECFP4 [%.1f,%.1f): admitted mean len %6.1f (n=%8d) | not-admitted %6.1f (n=%9d) | diff %+6.1f"
          % (b[0], b[1], A.mean(), A.size, Bn.mean(), Bn.size, A.mean() - Bn.mean()))
        sz.append(dict(bin_lo=b[0], bin_hi=b[1], mean_len_admitted=A.mean(), n_admitted=A.size,
                       mean_len_not=Bn.mean(), n_not=Bn.size, diff=A.mean() - Bn.mean()))
pd.DataFrame(sz).to_csv(os.path.join(C.OUT, "A6_size_gradient.csv"), index=False)
W("")
W("cliff rate by heavy-atom-count quartile (median across 30 datasets):")
W("  D0 (as shipped): " + "  ".join("Q%d=%.1f%%" % (i, d["D0_cliffrate_Q%d" % i].median()) for i in range(1, 5)))
W("  D2 (invariant):  " + "  ".join("Q%d=%.1f%%" % (i, d["D2_cliffrate_Q%d" % i].median()) for i in range(1, 5)))
open(os.path.join(C.OUT, "A6_composition.txt"), "w").write("\n".join(L))
