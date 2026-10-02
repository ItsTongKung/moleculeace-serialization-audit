"""Pooled analysis of the de-confounding arms (MACCS, PHYSCHEM) alongside ECFP4,
with dataset cluster-bootstrap CIs, for every cliff definition and every split arm.
"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

B = 4000
DATASETS = C.datasets()
MODELS = ['SVM', 'RF', 'GBM', 'KNN']
Z = {}
for f in sorted(glob.glob(os.path.join(C.PRED, "preds_alt", "*.npz"))
                + glob.glob(os.path.join(C.PRED, "preds_fullrepair", "*.npz"))):
    b = os.path.basename(f)[:-4]
    ds, arm, combo = b.split("__")
    model, desc = combo.split("_", 1)
    z = np.load(f, allow_pickle=True)
    Z[(ds, arm, model, desc)] = dict(se=(z['y_true'] - z['y_pred']) ** 2,
                                     **{k: z[k] for k in z.files if k.startswith('lab_')})
print("loaded", len(Z), "prediction files")


def sn(se, m):
    v = se[m]
    return float(v.sum()), int(v.size)


def pooled(D, keys):
    S = sum(D[k][0] for k in keys if k in D)
    N = sum(D[k][1] for k in keys if k in D)
    return np.sqrt(S / N) if N > 0 else np.nan


def boot(fn, seed=C.SEED):
    rng = np.random.default_rng(seed)
    p = fn(DATASETS)
    v = []
    for _ in range(B):
        s = list(rng.choice(DATASETS, size=len(DATASETS), replace=True))
        x = fn(s)
        if x is not None and np.isfinite(x):
            v.append(x)
    if len(v) < B // 10:
        return p, np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return p, float(lo), float(hi)


rows, L = [], []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)

W("=" * 110)
W("A-5  DE-CONFOUNDING: cliff penalty (own-complement) by DESCRIPTOR family")
W("     ECFP4 shares its representation with the D1/D2 cliff definitions; MACCS partly; PHYSCHEM not at all.")
W("=" * 110)
for arm in ['S0', 'R0', 'R2']:
    W("")
    W("  --- arm %s ---" % arm)
    for defn in ['D1', 'D2', 'D0', 'D3']:
        line = "   %-3s " % defn
        for desc in ['ECFP', 'MACCS', 'PHYSCHEM']:
            vals = []
            for model in MODELS:
                k = (arm, model, desc)
                A, Bd = {}, {}
                for ds in DATASETS:
                    kk = (ds, arm, model, desc)
                    if kk not in Z:
                        continue
                    lab = Z[kk]['lab_' + defn] == 1
                    if lab.sum() >= 1:
                        A[ds] = sn(Z[kk]['se'], lab)
                    if (~lab).sum() >= 1:
                        Bd[ds] = sn(Z[kk]['se'], ~lab)
                if not A:
                    continue
                p, lo, hi = boot(lambda ks: pooled(A, ks) - pooled(Bd, ks))
                vals.append(p)
                rows.append(dict(arm=arm, definition=defn, descriptor=desc, model=model,
                                 estimate=p, ci_lo=lo, ci_hi=hi, ci_status=C.ci_status(lo, hi)))
            if vals:
                sub = [r for r in rows if r['arm'] == arm and r['definition'] == defn
                       and r['descriptor'] == desc]
                nx = sum(1 for r in sub if r['ci_status'] == 'EXCLUDES 0')
                line += " %-8s %+.3f to %+.3f (%d/4 CI excl. 0) |" % (desc, min(vals), max(vals), nx)
        W(line)

pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "A5_deconfound_pooled.csv"), index=False)
open(os.path.join(C.OUT, "A5_deconfound_pooled.txt"), "w").write("\n".join(L))
print("\nwrote A5_deconfound_pooled.csv / .txt")
