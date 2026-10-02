"""A-2b  Randomised-SMILES invariance under PER-MOLECULE seeding.

`MolToRandomSmilesVect(mol, 1, randomSeed=s)` applies the same seed to every
molecule in a replicate, which correlates the atom-ordering choices across
molecules and slightly preserves string similarity.  A user who re-serialises a
dataset draws independently for each molecule.  This variant reproduces that
semantics deterministically by deriving a per-molecule seed
    seed_i = replicate_seed * 1000003 + i
so each molecule receives its own reproducible random stream.  Both variants are reported; neither is presented as the other.

The graph-criterion positive control is recomputed for the first replicate only
(it is exactly invariant by construction, and was already verified on 300
replicate-dataset combinations in 06_invariance.py).
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem

CACHE = os.path.join(C.OUT, "cache")
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
rows, perseed = [], []
nfail = 0
_rp = os.path.join(C.OUT, "A2b_invariance_permol.csv")
_pp = os.path.join(C.OUT, "A2b_invariance_permol_perseed.csv")
_done = set()
if os.path.exists(_rp):
    _d = pd.read_csv(_rp); rows = _d.to_dict('records'); _done = set(_d.dataset)
    if os.path.exists(_pp):
        perseed = pd.read_csv(_pp).to_dict('records')
    print("resuming: %d datasets already done" % len(_done), flush=True)
for k, ds in enumerate(C.datasets()):
    if ds in _done:
        print("[%2d/30] %-18s cached" % (k + 1, ds), flush=True); continue
    t0 = time.time()
    df = C.load(ds)
    smi = df['smiles'].tolist()
    n = len(smi)
    az = np.load(os.path.join(CACHE, "adj_" + ds + ".npz"))
    Tb = np.unpackbits(az['packed_Tb'], axis=1)[:, :n].astype(bool)
    Sb = np.unpackbits(az['packed_Sb'], axis=1)[:, :n].astype(bool)
    Fb = np.unpackbits(az['packed_Fb'], axis=1)[:, :n].astype(bool)
    Lb = np.unpackbits(az['packed_Lb'], axis=1)[:, :n].astype(bool)
    base = C.compound_labels(Tb | Sb | Lb, Fb)
    mols = [Chem.MolFromSmiles(s) for s in smi]
    ref_can = [Chem.MolToSmiles(m) for m in mols]
    flips, jacs, ncl = [], [], []
    ever = np.zeros(n, bool)
    for r, sd in enumerate(SEEDS):
        rs = [Chem.MolToRandomSmilesVect(m, 1, randomSeed=int(sd) * 1000003 + i)[0]
              for i, m in enumerate(mols)]
        bad = sum(1 for i, s in enumerate(rs)
                  if Chem.MolFromSmiles(s) is None
                  or Chem.MolToSmiles(Chem.MolFromSmiles(s)) != ref_can[i])
        assert bad == 0, "IDENTITY FAILURE %s seed=%d: %d molecules changed" % (ds, sd, bad)
        lab = C.compound_labels(Tb | Sb | (C.levenshtein_sim(rs) >= C.SIM), Fb)
        cT = cS = -1
        if r == 0:
            cT = int(((C.tanimoto(C.morgan(rs, False)) >= C.SIM) != Tb).sum())
            cS = int(((C.tanimoto(C.morgan(rs, True)) >= C.SIM) != Sb).sum())
            nfail += cT + cS
        d = (lab != base)
        flips.append(100 * d.mean()); ever |= d
        inter = int(np.logical_and(lab == 1, base == 1).sum())
        uni = int(np.logical_or(lab == 1, base == 1).sum())
        jacs.append(inter / uni if uni else np.nan)
        ncl.append(int(lab.sum()))
        perseed.append(dict(dataset=ds, seed=sd, flip_pct=100 * d.mean(),
                            n_cliff_base=int(base.sum()), n_cliff_rand=int(lab.sum()),
                            jaccard=jacs[-1], ctrl_ecfp_pair_diffs=cT, ctrl_scaffold_pair_diffs=cS))
    rows.append(dict(dataset=ds, n=n, n_cliff_base=int(base.sum()),
                     rand_flip_mean_pct=float(np.mean(flips)),
                     rand_flip_sd_pct=float(np.std(flips, ddof=1)),
                     rand_flip_min_pct=float(np.min(flips)), rand_flip_max_pct=float(np.max(flips)),
                     rand_jaccard_mean=float(np.mean(jacs)), rand_n_cliff_mean=float(np.mean(ncl)),
                     ever_flip_pct=100 * float(ever.mean()), secs=round(time.time() - t0, 1)))
    r_ = rows[-1]
    print("[%2d/30] %-18s n=%5d flip %5.2f+-%.2f%% (J=%.3f) [%.0fs]"
          % (k + 1, ds, n, r_['rand_flip_mean_pct'], r_['rand_flip_sd_pct'],
             r_['rand_jaccard_mean'], r_['secs']), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "A2b_invariance_permol.csv"), index=False)
    pd.DataFrame(perseed).to_csv(os.path.join(C.OUT, "A2b_invariance_permol_perseed.csv"), index=False)

o = pd.DataFrame(rows)
L = ["=" * 100,
     "A-2b  RANDOMISED SMILES, PER-MOLECULE SEEDING (seed_i = s*1000003 + i, s in 42..51)",
     "=" * 100,
     "positive control (ECFP4 + generic-scaffold pair diffs, first replicate of each dataset): %d (must be 0)" % nfail,
     "labels flipped median %.2f%% (range %.2f-%.2f%%)  compound-weighted mean %.2f%%"
     % (o.rand_flip_mean_pct.median(), o.rand_flip_min_pct.min(), o.rand_flip_max_pct.max(),
        (o.rand_flip_mean_pct * o.n).sum() / o.n.sum()),
     "Jaccard median %.3f (range %.3f-%.3f)"
     % (o.rand_jaccard_mean.median(), o.rand_jaccard_mean.min(), o.rand_jaccard_mean.max()),
     "per-dataset SD across the ten seeds: median %.3f pp, max %.3f pp"
     % (o.rand_flip_sd_pct.median(), o.rand_flip_sd_pct.max()),
     "total cliff compounds: shipped %d -> randomised mean %.0f"
     % (int(o.n_cliff_base.sum()), o.rand_n_cliff_mean.sum())]
print("\n".join(L))
open(os.path.join(C.OUT, "A2b_invariance_permol.txt"), "w").write("\n".join(L))
