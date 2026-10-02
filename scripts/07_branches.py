"""A-4  Similarity-branch operating regions.

MoleculeACE applies the same nominal 0.9 cut-off to three similarity measures
that are not numerically or chemically commensurate, and takes their
disjunction.  The scientific question is NOT "does a string-only pair reach
ECFP4 >= 0.9" (it cannot, by construction: reaching 0.9 on ECFP4 would admit it
under branch (a)).  The question is what effective structural region each
branch admits.  We therefore report, on a common yardstick (ECFP4 Tanimoto):

  ADMITTED  pairs the branch admits, regardless of what other branches do
  EXCLUSIVE pairs admitted ONLY by that branch
and we do this for the string branch AND the generic-scaffold branch, so the
string branch is not singled out without its proper comparator.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

CACHE = os.path.join(C.OUT, "cache")
KEYS = ['ecfp_admitted', 'scaf_admitted', 'str_admitted',
        'ecfp_only', 'scaf_only', 'str_only', 'all_cliffpairs']
pool = {k: [] for k in KEYS}
rows = []
for ds in C.datasets():
    z = np.load(os.path.join(CACHE, ds + ".npz"), allow_pickle=True)
    r = dict(dataset=ds)
    for k in KEYS:
        v = z["tsim_" + k]
        pool[k].append(v)
        r["n_" + k] = int(v.size)
        r["med_" + k] = float(np.median(v)) if v.size else np.nan
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(C.OUT, "A4_branch_perdataset.csv"), index=False)

P = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    P.append(s)

W("=" * 108)
W("A-4  EFFECTIVE OPERATING REGION OF EACH SIMILARITY BRANCH, MEASURED ON A COMMON YARDSTICK (ECFP4 TANIMOTO)")
W("=" * 108)
W("All rows are CLIFF PAIRS (similarity criterion satisfied AND >10-fold potency difference).")
W("")
hdr = "%-18s %9s %7s %7s %7s %7s %7s %7s %7s" % ("pair set", "n", "min", "p5", "q1", "median", "q3", "p95", "max")
W(hdr); W("-" * len(hdr))
stats = {}
for k in KEYS:
    v = np.concatenate(pool[k]) if pool[k] else np.array([])
    if v.size == 0:
        W("%-18s %9d   (empty)" % (k, 0)); continue
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    stats[k] = dict(n=int(v.size), **{a: float(b) for a, b in
                                      zip(['min', 'p5', 'q1', 'med', 'q3', 'p95', 'max'], q)})
    W("%-18s %9d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (k, v.size, *q))
W("")
tot = stats['all_cliffpairs']['n']
W("BRANCH SHARES OF THE %d CLIFF PAIRS:" % tot)
for k in ['ecfp_admitted', 'scaf_admitted', 'str_admitted']:
    W("  admitted by %-16s %7d (%5.1f%%)" % (k.split('_')[0], stats[k]['n'], 100 * stats[k]['n'] / tot))
for k in ['ecfp_only', 'scaf_only', 'str_only']:
    W("  ONLY      by %-16s %7d (%5.1f%%)" % (k.split('_')[0], stats[k]['n'], 100 * stats[k]['n'] / tot))
n_multi = tot - stats['ecfp_only']['n'] - stats['scaf_only']['n'] - stats['str_only']['n']
W("  admitted by >1 branch      %7d (%5.1f%%)" % (n_multi, 100 * n_multi / tot))
W("")
W("KEY NON-TAUTOLOGICAL COMPARISON --- the two non-fingerprint branches against each other:")
W("  generic-scaffold-only pairs: median ECFP4 %.3f (IQR %.3f-%.3f), %.1f%% below 0.5"
  % (stats['scaf_only']['med'], stats['scaf_only']['q1'], stats['scaf_only']['q3'],
     100 * float((np.concatenate(pool['scaf_only']) < 0.5).mean())))
W("  SMILES-string-only pairs:    median ECFP4 %.3f (IQR %.3f-%.3f), %.1f%% below 0.5"
  % (stats['str_only']['med'], stats['str_only']['q1'], stats['str_only']['q3'],
     100 * float((np.concatenate(pool['str_only']) < 0.5).mean())))
W("  ECFP4 branch (by construction >= 0.9): median %.3f" % stats['ecfp_admitted']['med'])
W("")
sv = np.concatenate(pool['str_only']); cv = np.concatenate(pool['scaf_only'])
W("  Neither non-fingerprint branch operates anywhere near 0.9 on the fingerprint scale.")
W("  string-only  spans %.3f-%.3f; scaffold-only spans %.3f-%.3f" % (sv.min(), sv.max(), cv.min(), cv.max()))
W("  pairs below ECFP4 0.5 admitted as 'similar at >= 0.9':  string branch %d, scaffold branch %d"
  % (int((sv < 0.5).sum()), int((cv < 0.5).sum())))
W("")
W("PER-DATASET medians (median across the 30 datasets):")
for k in ['ecfp_only', 'scaf_only', 'str_only']:
    W("  %-16s %.3f (range %.3f-%.3f over datasets)"
      % (k, d["med_" + k].median(), d["med_" + k].min(), d["med_" + k].max()))
open(os.path.join(C.OUT, "A4_branch_operating_regions.txt"), "w").write("\n".join(P))

# figure source data: histogram of ECFP4 similarity by branch
edges = np.arange(0, 1.0001, 0.025)
hist = pd.DataFrame({'bin_lo': edges[:-1], 'bin_hi': edges[1:]})
for k in ['ecfp_admitted', 'scaf_only', 'str_only']:
    v = np.concatenate(pool[k])
    hist[k] = np.histogram(v, bins=edges)[0]
hist.to_csv(os.path.join(C.OUT, "A4_branch_histogram.csv"), index=False)
print("\nwrote A4_branch_operating_regions.txt, A4_branch_perdataset.csv, A4_branch_histogram.csv")
