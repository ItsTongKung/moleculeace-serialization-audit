"""E3  DIRECT PAIRED BOOTSTRAP CONTRASTS.

Four comparisons that could otherwise only be judged by inspecting whether two
separately bootstrapped intervals overlap are here estimated as SINGLE paired quantities, with
the same resampled datasets entering every term of the contrast.  Same
dataset-cluster resampling logic as the primary contrast (B=4000, seed 42).

  A  (R2 - R0)  minus  (fixed-split relabelling: gap(S0,D2) - gap(S0,D0))
       "the two designs answer different questions" -- is the difference between
       the two designs itself resolvable?
  B  (R2 - R0)  minus  (R0 - S0)
       is the primary contrast resolvably larger than the split-drift control?
       (Note: the control remains a LOWER BOUND on split-realisation effect
       because its perturbation is smaller; 17_splitnoise.py supplies the
       magnitude-matched version.)
  C  within each model family: gap(D, ECFP4) - gap(D, MACCS)
  D  within each model family: gap(D, ECFP4) - gap(D, PHYSCHEM)
       Table 3's descriptor comparisons, as paired contrasts.
       Run for D = D1 (primary, the rung where the claim is made), and
       extended to D0 and D2 because the manuscript's Discussion makes a
       descriptor claim about the SHIPPED definition too.

Empty-subgroup bookkeeping: every bootstrap reports how many draws were
discarded because a resample contained no dataset contributing to a required
subgroup.  A contrast whose dropped-draw count is material is flagged.

Outputs: OUT/E8_paired.csv, OUT/E8_paired.txt
"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

PRED = os.path.join(C.PRED, "preds_fullrepair")
PALT = os.path.join(C.PRED, "preds_alt")
B = 4000
MODELS = ['SVM', 'RF', 'GBM', 'KNN']
DATASETS = C.datasets()

Z = {}
for f in sorted(glob.glob(os.path.join(PRED, "*.npz"))):
    ds, arm, combo = os.path.basename(f)[:-4].split("__")
    z = np.load(f, allow_pickle=True)
    Z[(ds, arm, combo.split("_")[0], 'ECFP')] = dict(
        se=(z['y_true'] - z['y_pred']) ** 2,
        **{k: z[k] for k in z.files if k.startswith('lab_')})
for f in sorted(glob.glob(os.path.join(PALT, "*.npz"))):
    ds, arm, combo = os.path.basename(f)[:-4].split("__")
    model, desc = combo.split("_", 1)
    z = np.load(f, allow_pickle=True)
    Z[(ds, arm, model, desc)] = dict(se=(z['y_true'] - z['y_pred']) ** 2,
                                     **{k: z[k] for k in z.files if k.startswith('lab_')})
print("loaded %d prediction sets (ECFP + MACCS + PHYSCHEM)" % len(Z))


def sn(se, mask):
    v = se[mask]
    return float(v.sum()), int(v.size)


def pooled(m, keys):
    S = sum(m[k][0] for k in keys if k in m)
    N = sum(m[k][1] for k in keys if k in m)
    return np.sqrt(S / N) if N > 0 else np.nan


def gap_fn(arm, model, defn, desc='ECFP'):
    """own-complement penalty as a function of a dataset key list."""
    A, Bd = {}, {}
    for ds in DATASETS:
        k = (ds, arm, model, desc)
        if k not in Z:
            continue
        lab = Z[k]['lab_' + defn] == 1
        if lab.sum() >= 1:
            A[ds] = sn(Z[k]['se'], lab)
        if (~lab).sum() >= 1:
            Bd[ds] = sn(Z[k]['se'], ~lab)
    return lambda ks: pooled(A, ks) - pooled(Bd, ks)


def boot(fn, seed=C.SEED):
    rng = np.random.default_rng(seed)
    point = fn(DATASETS)
    vals, dropped = [], 0
    for _ in range(B):
        s = list(rng.choice(DATASETS, size=len(DATASETS), replace=True))
        v = fn(s)
        if v is not None and np.isfinite(v):
            vals.append(v)
        else:
            dropped += 1
    if len(vals) < B // 10:
        return point, np.nan, np.nan, dropped, len(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), dropped, len(vals)


rows, LOG = [], []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def report(tag, label, model, fn, note=""):
    p, lo, hi, dr, nv = boot(fn)
    st = C.ci_status(lo, hi)
    flag = "  [!! %d/%d draws dropped]" % (dr, B) if dr > B * 0.01 else ""
    W("  %-52s %-5s  %+.4f  95%% CI [%+.4f, %+.4f]  %-11s%s"
      % (label, model, p, lo, hi, st, flag))
    rows.append(dict(analysis=tag, contrast=label, model=model, estimate=p,
                     ci_lo=lo, ci_hi=hi, ci_status=st, B=B, dropped_draws=dr,
                     valid_draws=nv, note=note))


W("=" * 122)
W("E3  DIRECT PAIRED BOOTSTRAP CONTRASTS  (B=%d, cluster bootstrap over the 30 datasets, seed %d)" % (B, C.SEED))
W("=" * 122)
W("Every contrast below is a single paired quantity: the same resampled datasets enter all terms.")
W("")

# ------------------------------------------------------------------ A
W("A  full repair (R2-R0)  MINUS  fixed-split relabelling (gap(S0,D2)-gap(S0,D0))")
W("   Tests the manuscript's claim that the two designs are not interchangeable.")
for m in MODELS:
    f_r2, f_r0 = gap_fn('R2', m, 'D2'), gap_fn('R0', m, 'D0')
    f_s2, f_s0 = gap_fn('S0', m, 'D2'), gap_fn('S0', m, 'D0')
    report('A_design_difference', "(R2-R0) - (fixed-split relabelling)", m,
           lambda ks, a=f_r2, b=f_r0, c=f_s2, e=f_s0: (a(ks) - b(ks)) - (c(ks) - e(ks)))
W("")

# ------------------------------------------------------------------ B
W("B  full repair (R2-R0)  MINUS  split-drift control (R0-S0)")
W("   The control's perturbation is smaller than the primary contrast's, so this tests only")
W("   whether the two reported quantities differ, NOT whether split noise explains the primary")
W("   contrast. E4 supplies the magnitude-matched control for that question.")
for m in MODELS:
    f_r2, f_r0, f_s0 = gap_fn('R2', m, 'D2'), gap_fn('R0', m, 'D0'), gap_fn('S0', m, 'D0')
    report('B_primary_minus_control', "(R2-R0) - (R0-S0)", m,
           lambda ks, a=f_r2, b=f_r0, c=f_s0: (a(ks) - b(ks)) - (b(ks) - c(ks)))
W("")

# ------------------------------------------------------------------ C, D
W("C/D  DESCRIPTOR-FAMILY CONTRASTS, paired within model family and arm")
W("   gap(definition, ECFP4) - gap(definition, MACCS)  and  - gap(definition, PHYSCHEM)")
for arm, defs in [('S0', ['D1', 'D0', 'D2']), ('R0', ['D1', 'D0']), ('R2', ['D1', 'D2', 'D0'])]:
    for defn in defs:
        W("")
        W("  --- arm %s, definition %s ---" % (arm, defn))
        for m in MODELS:
            fe = gap_fn(arm, m, defn, 'ECFP')
            fm = gap_fn(arm, m, defn, 'MACCS')
            fp = gap_fn(arm, m, defn, 'PHYSCHEM')
            report('C_ecfp_minus_maccs', "gap(%s,%s,ECFP4) - gap(MACCS)" % (arm, defn), m,
                   lambda ks, a=fe, b=fm: a(ks) - b(ks), note="%s/%s" % (arm, defn))
            report('D_ecfp_minus_physchem', "gap(%s,%s,ECFP4) - gap(PHYSCHEM)" % (arm, defn), m,
                   lambda ks, a=fe, b=fp: a(ks) - b(ks), note="%s/%s" % (arm, defn))

# ------------------------------------------------------------------ summary
W("")
W("=" * 122)
W("SUMMARY: contrasts whose 95% interval excludes zero")
W("=" * 122)
r = pd.DataFrame(rows)
for tag in r.analysis.unique():
    sub = r[r.analysis == tag]
    nex = int((sub.ci_status == "EXCLUDES 0").sum())
    W("  %-28s %3d of %3d intervals exclude zero" % (tag, nex, len(sub)))
W("")
W("Total draws dropped for empty subgroups: %d across %d bootstraps (max in any single bootstrap: %d/%d)"
  % (int(r.dropped_draws.sum()), len(r), int(r.dropped_draws.max()), B))

r.to_csv(os.path.join(C.OUT, "E8_paired.csv"), index=False)
open(os.path.join(C.OUT, "E8_paired.txt"), "w").write("\n".join(LOG))
print("\nwrote E8_paired.{csv,txt}")
