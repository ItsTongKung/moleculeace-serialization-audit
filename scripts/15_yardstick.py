"""E2  NON-BRANCH OPERATING-REGION YARDSTICK.

Fig. 1b and the sentence "the branch that departs furthest from the fingerprint
operating point is the anonymised-graph branch" are measured on ECFP4 Tanimoto.
That ruler is not neutral: ECFP4 IS branch (a), and it encodes element identity
and bond order -- exactly what the anonymised-graph branch (b) exists to
discard.  A pair that branch (b) uniquely admits is therefore *expected* to
score low on ECFP4 whether or not it is chemically unusual.

This script repeats the operating-region analysis on rulers that are NOT one of
the three MoleculeACE branches:

  RULER 1  MACCS-key Tanimoto (167 structural keys).  Element-aware like ECFP4
           but a different, substructure-dictionary representation; not a
           MoleculeACE branch.
  RULER 2  MCS-based similarity: maximum common substructure (RDKit rdFMCS,
           element- and bond-order-sensitive, ring-matching), scored as
           n_MCS_atoms / min(n_heavy_i, n_heavy_j).  Graph-native, not a
           fingerprint at all.  Computed on a pre-specified random subsample
           per branch category because rdFMCS is O(minutes) at full scale.
  (A third, branch-(b)-aligned diagnostic -- MCS on the element- and
  bond-order-anonymised graphs -- was specified and then DROPPED, for two
  two reasons: it is combinatorially intractable
  (all-carbon single-bond graphs are so symmetric that rdFMCS hits its timeout
  on most cliff pairs, costing ~4 min per dataset), and it is scientifically
  redundant, because a branch-(b)-exclusive pair satisfies anonymised-graph
  Tanimoto >= 0.9 BY CONSTRUCTION -- that is what admitted it.  The caveat it
  was meant to support therefore needs no computation and is stated directly:
  every ruler tested here retains element identity and bond order, which
  branch (b) exists to discard, so all of them are expected to place
  branch-(b)-exclusive pairs low, and the test below is whether the ORDERING
  against the string branch survives, not whether the absolute level is fair.)

PRE-SPECIFIED DECISION RULE (fixed before running):
  the manuscript claim under test is the ORDERING
      median(similarity | anonymised-graph-only) < median(similarity | string-only)
  i.e. the anonymised-graph branch sits farther from the fingerprint-like
  operating region than the string branch.
  If the ordering holds on the alternative rulers -> retain the attribution,
  scoped to element-aware rulers.
  If it reverses or becomes indistinguishable -> DELETE every "furthest
  departure" statement from Abstract, Results, Discussion and Conclusions.
  The separate claim that the three nominal 0.9 thresholds are not commensurate
  across criteria does not depend on this test and may stand either way.

Inference: the ordering difference is bootstrapped as a PAIRED quantity over
the 30 datasets (cluster bootstrap, B=4000, seed 42) -- the per-dataset median
for both categories comes from the same resampled datasets.

Outputs: OUT/E7_yardstick.csv (per dataset x ruler x category),
         OUT/E7_yardstick_pairs.csv (pooled percentiles),
         OUT/E7_yardstick_hist.csv (figure source data),
         OUT/E7_yardstick.txt
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs, rdFMCS
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

CACHE = os.path.join(C.OUT, "cache")
B = 4000
MCS_SUBSAMPLE = 120          # pairs per (dataset, category); fixed before the run for wall-clock reasons; MACCS uses the full pair set
MCS_TIMEOUT = 1              # seconds per MCS call
RNG_SEED = C.SEED
CATS = ['ecfp_admitted', 'scaf_only', 'str_only']


def maccs_matrix(smiles):
    fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)) for s in smiles]
    n = len(fps)
    M = np.zeros((n, n), np.float32)
    for i in range(n):
        M[i, i:] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i:])
    M = np.maximum(M, M.T)
    np.fill_diagonal(M, 0.0)
    return M


def anon(m):
    try:
        return GraphFramework(m)
    except Exception:
        return GetScaffoldForMol(m)


def mcs_sim(a, b):
    """n_MCS_atoms / min(heavy_a, heavy_b); 0.0 if no MCS found in time."""
    if a is None or b is None:
        return np.nan
    try:
        r = rdFMCS.FindMCS([a, b], timeout=MCS_TIMEOUT,
                           ringMatchesRingOnly=True, completeRingsOnly=False,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    except Exception:
        return np.nan
    d = min(a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms())
    if d == 0:
        return np.nan
    return float(r.numAtoms) / d


rows, pool = [], {(r, c): [] for r in ['maccs', 'mcs'] for c in CATS}
POOLDIR = os.path.join(C.OUT, "e7pool"); os.makedirs(POOLDIR, exist_ok=True)
_rp = os.path.join(C.OUT, "E7_yardstick.csv")
_done = set()
if os.path.exists(_rp):
    _d = pd.read_csv(_rp); rows = _d.to_dict('records'); _done = set(_d.dataset)
    print("resuming: %d datasets already done" % len(_done), flush=True)
t_start = time.time()
for k, ds in enumerate(C.datasets()):
    if ds in _done:
        print("[%2d/30] %-18s cached" % (k + 1, ds), flush=True)
        _z = np.load(os.path.join(POOLDIR, ds + ".npz"))
        for _r in ['maccs', 'mcs']:
            for _c in CATS:
                pool[(_r, _c)].append(_z["%s__%s" % (_r, _c)])
        continue
    t0 = time.time()
    df = C.load(ds)
    smi = df['smiles'].tolist()
    n = len(smi)
    az = np.load(os.path.join(CACHE, "adj_%s.npz" % ds))
    Tb = np.unpackbits(az['packed_Tb'], axis=1)[:, :n].astype(bool)
    Sb = np.unpackbits(az['packed_Sb'], axis=1)[:, :n].astype(bool)
    Lb = np.unpackbits(az['packed_Lb'], axis=1)[:, :n].astype(bool)
    Fb = np.unpackbits(az['packed_Fb'], axis=1)[:, :n].astype(bool)

    iu = np.triu_indices(n, 1)
    cliffpair = Fb[iu] & (Tb | Sb | Lb)[iu]
    t_i, s_i, l_i = Tb[iu], Sb[iu], Lb[iu]
    masks = {'ecfp_admitted': cliffpair & t_i,
             'scaf_only':     cliffpair & s_i & ~t_i & ~l_i,
             'str_only':      cliffpair & l_i & ~t_i & ~s_i}

    Mm = maccs_matrix(smi)
    mv = Mm[iu]

    mols = [Chem.MolFromSmiles(s) for s in smi]
    rng = np.random.default_rng(RNG_SEED + k)

    r = dict(dataset=ds, n=n)
    for cat in CATS:
        m = masks[cat]
        vals = mv[m].astype(np.float32)
        pool[('maccs', cat)].append(vals)
        r["n_" + cat] = int(m.sum())
        r["maccs_med_" + cat] = float(np.median(vals)) if vals.size else np.nan
        r["maccs_below05_pct_" + cat] = 100.0 * float((vals < 0.5).mean()) if vals.size else np.nan

        idx = np.where(m)[0]
        if idx.size > MCS_SUBSAMPLE:
            idx = rng.choice(idx, MCS_SUBSAMPLE, replace=False)
        ii, jj = iu[0][idx], iu[1][idx]
        sv = [mcs_sim(mols[a], mols[b]) for a, b in zip(ii, jj)]
        sv = np.array([x for x in sv if np.isfinite(x)], np.float32)
        pool[('mcs', cat)].append(sv)
        r["mcs_n_" + cat] = int(sv.size)
        r["mcs_med_" + cat] = float(np.median(sv)) if sv.size else np.nan
    r['secs'] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(POOLDIR, ds + ".npz"),
                        **{"%s__%s" % (rr, cc): pool[(rr, cc)][-1]
                           for rr in ['maccs', 'mcs'] for cc in CATS})
    rows.append(r)
    print("[%2d/30] %-18s MACCS med  ecfp=%.3f scaf_only=%.3f str_only=%.3f | "
          "MCS med scaf_only=%.3f str_only=%.3f  [%.0fs]"
          % (k + 1, ds, r['maccs_med_ecfp_admitted'], r['maccs_med_scaf_only'],
             r['maccs_med_str_only'],
             r['mcs_med_scaf_only'] if np.isfinite(r['mcs_med_scaf_only']) else float('nan'),
             r['mcs_med_str_only'] if np.isfinite(r['mcs_med_str_only']) else float('nan'),
             r['secs']), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "E7_yardstick.csv"), index=False)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(C.OUT, "E7_yardstick.csv"), index=False)

LOG = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

W("=" * 118)
W("E2  OPERATING REGIONS UNDER RULERS THAT ARE NOT MoleculeACE BRANCHES")
W("=" * 118)
W("MCS subsample: up to %d pairs per (dataset, category), seed %d, rdFMCS timeout %ds,"
  % (MCS_SUBSAMPLE, RNG_SEED, MCS_TIMEOUT))
W("               CompareElements / CompareOrderExact, ringMatchesRingOnly=True;")
W("               score = n_MCS_atoms / min(heavy_i, heavy_j).")
W("")
percs = []
for ruler in ['maccs', 'mcs']:
    W("--- RULER: %s ---" % ruler.upper())
    hdr = "%-16s %9s %7s %7s %7s %7s %7s" % ("category", "n", "q1", "median", "q3", "<0.5%", "min-max")
    W(hdr); W("-" * len(hdr))
    for cat in CATS:
        v = np.concatenate(pool[(ruler, cat)]) if pool[(ruler, cat)] else np.array([])
        if v.size == 0:
            W("%-16s %9d  (empty)" % (cat, 0)); continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        W("%-16s %9d %7.3f %7.3f %7.3f %6.1f%% %.3f-%.3f"
          % (cat, v.size, q1, med, q3, 100 * float((v < 0.5).mean()), v.min(), v.max()))
        percs.append(dict(ruler=ruler, category=cat, n=int(v.size), q1=q1, median=med, q3=q3,
                          pct_below_0p5=100 * float((v < 0.5).mean()),
                          min=float(v.min()), max=float(v.max())))
    W("")
pd.DataFrame(percs).to_csv(os.path.join(C.OUT, "E7_yardstick_pairs.csv"), index=False)

# ------------------------------------------------- paired cluster bootstrap of the ordering
W("=" * 118)
W("ORDERING TEST  delta = median(scaf_only) - median(str_only), per ruler")
W("  delta < 0  => anonymised-graph branch sits FARTHER from the fingerprint-like operating region (claim HOLDS)")
W("  delta > 0  => ordering REVERSES (claim must be deleted)")
W("  Paired cluster bootstrap over the 30 datasets, B=%d, seed %d." % (B, RNG_SEED))
W("=" * 118)
DS = list(d.dataset)
idx = {ds: i for i, ds in enumerate(DS)}
order_rows = []
for ruler in ['maccs', 'mcs']:
    A = {ds: pool[(ruler, 'scaf_only')][idx[ds]] for ds in DS}
    Bp = {ds: pool[(ruler, 'str_only')][idx[ds]] for ds in DS}

    def stat(keys):
        a = np.concatenate([A[k] for k in keys if A[k].size])
        b = np.concatenate([Bp[k] for k in keys if Bp[k].size])
        if a.size == 0 or b.size == 0:
            return np.nan
        return float(np.median(a) - np.median(b))

    point = stat(DS)
    rng = np.random.default_rng(RNG_SEED)
    vals, dropped = [], 0
    for _ in range(B):
        s = list(rng.choice(DS, size=len(DS), replace=True))
        v = stat(s)
        if np.isfinite(v):
            vals.append(v)
        else:
            dropped += 1
    lo, hi = np.percentile(vals, [2.5, 97.5])
    verdict = ("HOLDS (delta<0, interval excludes 0)" if hi < 0 else
               "REVERSES (delta>0, interval excludes 0)" if lo > 0 else
               "NOT RESOLVED (interval contains 0)")
    W("  %-9s delta = %+.4f  95%% CI [%+.4f, %+.4f]  dropped draws %d  -> %s"
      % (ruler, point, lo, hi, dropped, verdict))
    order_rows.append(dict(ruler=ruler, delta_median=point, ci_lo=lo, ci_hi=hi,
                           B=B, dropped_draws=dropped, verdict=verdict,
                           ci_status=C.ci_status(lo, hi)))
    # per-dataset sign count (equal dataset weighting, secondary)
    sgn = [(np.median(A[ds]) - np.median(Bp[ds])) for ds in DS
           if A[ds].size and Bp[ds].size]
    W("            per-dataset (equal weighting): negative in %d/%d datasets, mean %+.4f, median %+.4f"
      % (int(sum(1 for x in sgn if x < 0)), len(sgn), float(np.mean(sgn)), float(np.median(sgn))))
pd.DataFrame(order_rows).to_csv(os.path.join(C.OUT, "E7_yardstick_ordering.csv"), index=False)

W("")
W("NOTE: every ruler tested retains element identity and bond order, which branch (b) exists to discard,")
W("so all of them are expected to place branch-(b)-exclusive pairs low. The test is whether the ORDERING")
W("against the string branch survives, not whether the absolute level is fair to branch (b).")

# ------------------------------------------------- figure source data
edges = np.arange(0, 1.0001, 0.025)
hist = pd.DataFrame({'bin_lo': edges[:-1], 'bin_hi': edges[1:]})
for ruler in ['maccs', 'mcs']:
    for cat in CATS:
        v = np.concatenate(pool[(ruler, cat)])
        hist["%s_%s" % (ruler, cat)] = np.histogram(v, bins=edges)[0]
hist.to_csv(os.path.join(C.OUT, "E7_yardstick_hist.csv"), index=False)
W("")
W("total wall time %.1f min" % ((time.time() - t_start) / 60))
open(os.path.join(C.OUT, "E7_yardstick.txt"), "w").write("\n".join(LOG))
print("\nwrote E7_yardstick.{csv,txt}, E7_yardstick_pairs.csv, E7_yardstick_ordering.csv, E7_yardstick_hist.csv")
