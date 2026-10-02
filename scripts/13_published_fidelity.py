"""E5  Fidelity of our re-run against the benchmark's own distributed results.

Compares our S0-arm (shipped split, published hyperparameters, ECFP4-1024) RMSE
and D0 cliff-RMSE with `Data/results/MoleculeACE_results.csv` as distributed.
Also checks the identity between the randomised-SMILES flip rate and the
D0 -> D2 relabelling rate.
"""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

pub = pd.read_csv(os.path.join(os.path.dirname(C.DATA), "results", "MoleculeACE_results.csv"))
pub = pub[(pub.descriptor == 'ECFP') & (pub.augmentation == 0)]
ours = pd.read_csv(os.path.join(C.OUT, "A1_fullrepair_results.csv"))
ours = ours[ours.arm == 'S0']
L = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)

W("=" * 104)
W("E5  FIDELITY AGAINST THE DISTRIBUTED MoleculeACE RESULTS (ECFP arm, augmentation 0)")
W("=" * 104)
W("%-6s %10s %10s %10s | %12s %10s %10s" % ("model", "n", "r(RMSE)", "mean|d|", "r(cliffRMSE)", "mean|d|", "our-pub"))
rows = []
for m in ['SVM', 'RF', 'GBM', 'KNN']:
    p = pub[pub.algorithm == m].set_index('dataset')
    o = ours[ours.model == m].set_index('dataset')
    j = o.join(p, how='inner', rsuffix='_pub')
    r1 = st.pearsonr(j.rmse_all, j.rmse).statistic
    d1 = float(np.abs(j.rmse_all - j.rmse).mean())
    r2 = st.pearsonr(j.rmse_D0, j.cliff_rmse).statistic
    d2 = float(np.abs(j.rmse_D0 - j.cliff_rmse).mean())
    W("%-6s %10d %10.4f %10.4f | %12.4f %10.4f %10.4f"
      % (m, len(j), r1, d1, r2, d2, float((j.rmse_D0 - j.cliff_rmse).mean())))
    rows.append(dict(model=m, n=len(j), pearson_rmse=r1, mad_rmse=d1,
                     pearson_cliffrmse=r2, mad_cliffrmse=d2))
pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "E5_published_fidelity.csv"), index=False)

W("")
W("=" * 104)
W("E6  IS THE RANDOMISED-SMILES FLIP RATE THE SAME MEASUREMENT AS THE D0 -> D2 RELABELLING RATE?")
W("=" * 104)
comp = pd.read_csv(os.path.join(C.OUT, "A6_composition.csv")).set_index('dataset')
for tag, f in [("shared-seed", "A2_invariance.csv"), ("per-molecule-seed", "A2b_invariance_permol.csv")]:
    p = os.path.join(C.OUT, f)
    if not os.path.exists(p):
        continue
    inv = pd.read_csv(p).set_index('dataset')
    j = inv.join(comp[['pct_relabelled_D0_to_D2']])
    r = st.pearsonr(j.rand_flip_mean_pct, j.pct_relabelled_D0_to_D2).statistic
    mad = float(np.abs(j.rand_flip_mean_pct - j.pct_relabelled_D0_to_D2).mean())
    W("  %-18s Pearson r = %.5f | mean |difference| = %.3f pp | medians %.2f%% vs %.2f%%"
      % (tag, r, mad, j.rand_flip_mean_pct.median(), j.pct_relabelled_D0_to_D2.median()))
open(os.path.join(C.OUT, "E5_published_fidelity.txt"), "w").write("\n".join(L))
