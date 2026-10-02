"""A-1/A-3  Primary statistical analysis of the full-repair counterfactual,
the fixed-split relabelling analysis, the definitional ladder, and the
partner-location interaction test.

Inference throughout: pooled squared errors across datasets (all targets share
a log-potency scale), 95% CI from a CLUSTER BOOTSTRAP over datasets (B = 4000,
seed 42).  Differences between conditions are bootstrapped DIRECTLY and PAIRED
(the same resampled datasets enter both conditions), never by inspecting
whether two intervals overlap.

Pooled RMSE over a resample is computed as sqrt(sum SS_d / sum N_d), which is
algebraically identical to concatenating the errors but O(30) per draw.
"""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

PRED = os.path.join(C.PRED, "preds_fullrepair")
CACHE = os.path.join(C.OUT, "cache")
B = 4000
RNG = np.random.default_rng(C.SEED)
DEFS = ['D0', 'D1', 'D2', 'D3']
ARMS = ['S0', 'R0', 'R2']
MODELS = ['SVM', 'RF', 'GBM', 'KNN']
DATASETS = C.datasets()

# ---------------------------------------------------------------- load
Z = {}
for f in sorted(glob.glob(os.path.join(PRED, "*.npz"))):
    b = os.path.basename(f)[:-4]
    ds, arm, combo = b.split("__")
    model = combo.split("_")[0]
    z = np.load(f, allow_pickle=True)
    Z[(ds, arm, model)] = dict(se=(z['y_true'] - z['y_pred']) ** 2,
                               **{k: z[k] for k in z.files
                                  if k.startswith('lab_') or k.startswith('partner_')
                                  or k == 'nn_train'})
print("loaded %d prediction files" % len(Z))

# chemistry-free null predictor: per-dataset training mean, under each arm
NULL = {}
for ds in DATASETS:
    zc = np.load(os.path.join(CACHE, ds + ".npz"), allow_pickle=True)
    y = zc['y']
    for arm, key in [('S0', 'shipped_split'), ('R0', 'split_reconD0'), ('R2', 'split_reconD2')]:
        sp = zc[key].astype(str)
        tr, te = sp == 'train', sp == 'test'
        NULL[(ds, arm)] = (y[te] - y[tr].mean()) ** 2


def sn(se, mask):
    """(sum of squared errors, n) for a boolean mask; (0,0) if empty."""
    if mask is None:
        return float(se.sum()), int(se.size)
    v = se[mask]
    return float(v.sum()), int(v.size)


def pooled(sn_by_ds, keys):
    S = sum(sn_by_ds[k][0] for k in keys if k in sn_by_ds)
    N = sum(sn_by_ds[k][1] for k in keys if k in sn_by_ds)
    return np.sqrt(S / N) if N > 0 else np.nan


def boot(fn, B=B, seed=C.SEED):
    """fn(list_of_datasets) -> scalar.  Returns (point, lo, hi, n_valid)."""
    rng = np.random.default_rng(seed)
    point = fn(DATASETS)
    vals = []
    for _ in range(B):
        s = list(rng.choice(DATASETS, size=len(DATASETS), replace=True))
        v = fn(s)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < B // 10:
        return point, np.nan, np.nan, len(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), len(vals)


def gap_fn(arm, model, defn, source=None):
    """own-complement penalty: RMSE(cliff under defn) - RMSE(not-cliff under defn)"""
    A, Bd = {}, {}
    for ds in DATASETS:
        key = (ds, arm, model)
        if key not in Z:
            continue
        se = NULL[(ds, arm)] if source == 'null' else Z[key]['se']
        lab = Z[key]['lab_' + defn] == 1
        if lab.sum() >= 1:
            A[ds] = sn(se, lab)
        if (~lab).sum() >= 1:
            Bd[ds] = sn(se, ~lab)
    return lambda ks: pooled(A, ks) - pooled(Bd, ks)


def n_cliff(arm, model, defn):
    return int(sum((Z[(ds, arm, model)]['lab_' + defn] == 1).sum()
                   for ds in DATASETS if (ds, arm, model) in Z))


rows, LOG = [], []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ================================================================ 1. arms
W("=" * 118)
W("A-1a  HEADLINE CLIFF PENALTY BY ARM  (own-complement; arm's own cliff definition)")
W("      S0 = shipped split, D0 labels   R0 = split reconstructed from D0   R2 = split reconstructed from D2, retrained")
W("=" * 118)
W("%-5s %-5s %-4s %8s %8s %10s %10s %22s" % ("arm", "model", "def", "n_cliff", "n_test", "RMSE_all", "penalty", "95% CI"))
armgap = {}
for model in MODELS:
    for arm in ARMS:
        defn = 'D2' if arm == 'R2' else 'D0'
        f = gap_fn(arm, model, defn)
        p, lo, hi, _ = boot(f)
        armgap[(arm, model)] = f
        nt = int(sum(Z[(ds, arm, model)]['se'].size for ds in DATASETS if (ds, arm, model) in Z))
        allf = lambda ks, a=arm, m=model: pooled({ds: sn(Z[(ds, a, m)]['se'], None)
                                                  for ds in DATASETS if (ds, a, m) in Z}, ks)
        W("%-5s %-5s %-4s %8d %8d %10.4f %+10.4f   [%+.4f, %+.4f] %s"
          % (arm, model, defn, n_cliff(arm, model, defn), nt, allf(DATASETS), p, lo, hi, C.ci_status(lo, hi)))
        rows.append(dict(analysis='arm_penalty', arm=arm, model=model, definition=defn,
                         n_cliff=n_cliff(arm, model, defn), n_test=nt,
                         rmse_all=allf(DATASETS), estimate=p, ci_lo=lo, ci_hi=hi,
                         ci_status=C.ci_status(lo, hi)))

# ================================================================ 2. contrasts
W("")
W("=" * 118)
W("A-1b  PAIRED CONTRASTS  (bootstrapped directly; same resampled datasets in both conditions)")
W("=" * 118)
CONTRASTS = [
    ("PRIMARY  full repair      R2 - R0", 'R2', 'D2', 'R0', 'D0'),
    ("CONTROL  split drift      R0 - S0", 'R0', 'D0', 'S0', 'D0'),
    ("BRIDGE   shipped->repair  R2 - S0", 'R2', 'D2', 'S0', 'D0'),
]
for name, a1, d1, a0, d0 in CONTRASTS:
    for model in MODELS:
        f1, f0 = gap_fn(a1, model, d1), gap_fn(a0, model, d0)
        p, lo, hi, _ = boot(lambda ks: f1(ks) - f0(ks))
        W("%-36s %-5s  delta = %+.4f  95%% CI [%+.4f, %+.4f]  %s"
          % (name, model, p, lo, hi, C.ci_status(lo, hi)))
        rows.append(dict(analysis='contrast', contrast=name, model=model,
                         estimate=p, ci_lo=lo, ci_hi=hi, ci_status=C.ci_status(lo, hi)))
    W("")

W("=" * 118)
W("A-1c  OVERALL TEST RMSE BY ARM  (does repairing the benchmark change how hard it is overall?)")
W("=" * 118)
for model in MODELS:
    def allf(arm, m=model):
        d = {ds: sn(Z[(ds, arm, m)]['se'], None) for ds in DATASETS if (ds, arm, m) in Z}
        return lambda ks: pooled(d, ks)
    fS, fR0, fR2 = allf('S0'), allf('R0'), allf('R2')
    p1, l1, h1, _ = boot(lambda ks: fR2(ks) - fR0(ks))
    p2, l2, h2, _ = boot(lambda ks: fR0(ks) - fS(ks))
    W("%-5s RMSE_all  S0=%.4f  R0=%.4f  R2=%.4f | R2-R0 = %+.4f [%+.4f,%+.4f] %s | R0-S0 = %+.4f [%+.4f,%+.4f] %s"
      % (model, fS(DATASETS), fR0(DATASETS), fR2(DATASETS), p1, l1, h1, C.ci_status(l1, h1),
         p2, l2, h2, C.ci_status(l2, h2)))
    rows.append(dict(analysis='rmse_all', model=model, rmse_S0=fS(DATASETS), rmse_R0=fR0(DATASETS),
                     rmse_R2=fR2(DATASETS), estimate=p1, ci_lo=l1, ci_hi=h1,
                     ci_status=C.ci_status(l1, h1), contrast='R2-R0'))

# ================================================================ 3. fixed-split relabelling (the OLD estimand)
W("")
W("=" * 118)
W("A-1d  FIXED-SPLIT RELABELLING  (shipped split and shipped predictions, re-scored under D2)")
W("      This is the estimand of the previous draft.  It is NOT the full-repair estimand above.")
W("=" * 118)
wil = {}
for model in MODELS:
    f0, f2 = gap_fn('S0', model, 'D0'), gap_fn('S0', model, 'D2')
    p, lo, hi, _ = boot(lambda ks: f2(ks) - f0(ks))
    W("%-5s  gap(D0)=%+.4f  gap(D2)=%+.4f  delta = %+.4f  95%% CI [%+.4f, %+.4f]  %s"
      % (model, f0(DATASETS), f2(DATASETS), p, lo, hi, C.ci_status(lo, hi)))
    rows.append(dict(analysis='fixed_split_relabel', model=model, gap_D0=f0(DATASETS),
                     gap_D2=f2(DATASETS), estimate=p, ci_lo=lo, ci_hi=hi,
                     ci_status=C.ci_status(lo, hi)))
    # per-dataset paired Wilcoxon (secondary)
    a, b_ = [], []
    for ds in DATASETS:
        k = (ds, 'S0', model)
        if k not in Z:
            continue
        se = Z[k]['se']
        for d, acc in [('D0', a), ('D2', b_)]:
            m = Z[k]['lab_' + d] == 1
            if m.sum() >= 5 and (~m).sum() >= 5:
                acc.append(np.sqrt(se[m].mean()) - np.sqrt(se[~m].mean()))
            else:
                acc.append(np.nan)
    a, b_ = np.array(a), np.array(b_)
    ok = ~(np.isnan(a) | np.isnan(b_))
    wil[model] = st.wilcoxon(a[ok], b_[ok]).pvalue
pv = np.array([wil[m] for m in MODELS])
order = np.argsort(pv); holm = np.empty_like(pv); run = 0
for r, i in enumerate(order):
    run = max(run, (len(pv) - r) * pv[i]); holm[i] = min(run, 1)
W("  secondary per-dataset Wilcoxon (D0 vs D2 gaps, shipped split): "
  + ", ".join("%s p=%.4f (Holm %.3f)" % (m, wil[m], holm[i]) for i, m in enumerate(MODELS))
  + " | surviving Holm: %d/4" % int((holm < 0.05).sum()))
rows.append(dict(analysis='wilcoxon_holm', **{('p_' + m): wil[m] for m in MODELS},
                 **{('holm_' + m): holm[i] for i, m in enumerate(MODELS)}))

# ================================================================ 4. definitional ladder
W("")
W("=" * 118)
W("A-1e  DEFINITIONAL LADDER on the shipped benchmark (arm S0, own-complement baselines)")
W("=" * 118)
for model in MODELS:
    line = "%-5s " % model
    for d in ['D3', 'D0', 'D2', 'D1']:
        f = gap_fn('S0', model, d)
        p, lo, hi, _ = boot(f)
        line += " %s: %+.3f [%+.3f,%+.3f] (n=%d) " % (d, p, lo, hi, n_cliff('S0', model, d))
        rows.append(dict(analysis='ladder', arm='S0', model=model, definition=d,
                         n_cliff=n_cliff('S0', model, d), estimate=p, ci_lo=lo, ci_hi=hi,
                         ci_status=C.ci_status(lo, hi)))
    W(line)
W("  ratio decomposition (SVM..KNN): D1/D0 total, string step D0->D2, scaffold step D2->D1")
for model in MODELS:
    f0, f1, f2 = gap_fn('S0', model, 'D0'), gap_fn('S0', model, 'D1'), gap_fn('S0', model, 'D2')
    r_tot, lt, ht, _ = boot(lambda ks: f1(ks) / f0(ks))
    r_str, ls, hs, _ = boot(lambda ks: f2(ks) / f0(ks))
    r_sca, lc, hc, _ = boot(lambda ks: f1(ks) / f2(ks))
    W("   %-5s total %.2fx [%.2f,%.2f] | string step %.2fx [%.2f,%.2f] | scaffold step %.2fx [%.2f,%.2f]"
      % (model, r_tot, lt, ht, r_str, ls, hs, r_sca, lc, hc))
    rows.append(dict(analysis='ladder_ratio', model=model, ratio_total=r_tot, rt_lo=lt, rt_hi=ht,
                     ratio_string=r_str, rs_lo=ls, rs_hi=hs, ratio_scaffold=r_sca, rc_lo=lc, rc_hi=hc))

# ================================================================ 5. partner location + INTERACTION
W("")
W("=" * 118)
W("A-3  PARTNER LOCATION: is the penalty concentrated among compounds whose >10-fold partner is in TRAIN?")
W("     Reported: each subgroup vs the definition's non-cliff complement, AND the direct")
W("     subgroup DIFFERENCE (interaction), bootstrapped as a single paired quantity.")
W("=" * 118)
for defn in ['D1', 'D2', 'D0']:
    for arm in ['S0', 'R0'] if defn != 'D2' else ['R2']:
        nIN = int(sum(((Z[(ds, arm, 'SVM')]['lab_' + defn] == 1) &
                       (Z[(ds, arm, 'SVM')]['partner_train_' + defn])).sum()
                      for ds in DATASETS if (ds, arm, 'SVM') in Z))
        nOUT = int(sum(((Z[(ds, arm, 'SVM')]['lab_' + defn] == 1) &
                        (~Z[(ds, arm, 'SVM')]['partner_train_' + defn])).sum()
                       for ds in DATASETS if (ds, arm, 'SVM') in Z))
        W("")
        W("  --- definition %s, arm %s --- cliff test compounds: partner IN train n=%d (%.1f%%), NOT in train n=%d (%.1f%%)"
          % (defn, arm, nIN, 100 * nIN / max(nIN + nOUT, 1), nOUT, 100 * nOUT / max(nIN + nOUT, 1)))
        for model in MODELS:
            def sub(flag, m=model, a=arm, d=defn):
                A = {}
                for ds in DATASETS:
                    k = (ds, a, m)
                    if k not in Z:
                        continue
                    lab = Z[k]['lab_' + d] == 1
                    pt = Z[k]['partner_train_' + d]
                    mask = lab & (pt if flag else ~pt)
                    if mask.sum() >= 1:
                        A[ds] = sn(Z[k]['se'], mask)
                return A
            def nonc(m=model, a=arm, d=defn):
                A = {}
                for ds in DATASETS:
                    k = (ds, a, m)
                    if k not in Z:
                        continue
                    mask = Z[k]['lab_' + d] == 0
                    if mask.sum() >= 1:
                        A[ds] = sn(Z[k]['se'], mask)
                return A
            AIN, AOUT, NC = sub(True), sub(False), nonc()
            gIN = lambda ks: pooled(AIN, ks) - pooled(NC, ks)
            gOUT = lambda ks: pooled(AOUT, ks) - pooled(NC, ks)
            p1, l1, h1, _ = boot(gIN)
            p0, l0, h0, _ = boot(gOUT)
            pd_, ld, hd, _ = boot(lambda ks: gIN(ks) - gOUT(ks))
            W("    %-5s partner IN %+.4f [%+.4f,%+.4f] %-10s | NOT in %+.4f [%+.4f,%+.4f] %-10s | "
              "INTERACTION %+.4f [%+.4f,%+.4f] %s"
              % (model, p1, l1, h1, C.ci_status(l1, h1), p0, l0, h0, C.ci_status(l0, h0),
                 pd_, ld, hd, C.ci_status(ld, hd)))
            rows.append(dict(analysis='partner_location', arm=arm, model=model, definition=defn,
                             n_partner_in=nIN, n_partner_out=nOUT,
                             gap_in=p1, in_lo=l1, in_hi=h1, in_status=C.ci_status(l1, h1),
                             gap_out=p0, out_lo=l0, out_hi=h0, out_status=C.ci_status(l0, h0),
                             estimate=pd_, ci_lo=ld, ci_hi=hd, ci_status=C.ci_status(ld, hd)))

# ================================================================ 6. proximity strata
W("")
W("=" * 118)
W("A-3b  PROXIMITY-MATCHED CONTROL (strata of nearest-neighbour ECFP4 similarity to the training set), arm S0")
W("=" * 118)
allnn_c, allnn_n = [], []
for ds in DATASETS:
    k = (ds, 'S0', 'SVM')
    if k in Z:
        lab = Z[k]['lab_D1'] == 1
        allnn_c.append(Z[k]['nn_train'][lab]); allnn_n.append(Z[k]['nn_train'][~lab])
W("  nn_train median: D1 cliffs %.3f vs non-cliffs %.3f"
  % (np.median(np.concatenate(allnn_c)), np.median(np.concatenate(allnn_n))))
STRATA = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
for model in ['SVM', 'RF']:
    for lo_, hi_ in STRATA:
        A, Bd, n1, n0 = {}, {}, 0, 0
        for ds in DATASETS:
            k = (ds, 'S0', model)
            if k not in Z:
                continue
            nn = Z[k]['nn_train']; instr = (nn >= lo_) & (nn < hi_)
            lab = Z[k]['lab_D1'] == 1
            m1, m0 = instr & lab, instr & ~lab
            n1 += int(m1.sum()); n0 += int(m0.sum())
            if m1.sum() >= 1:
                A[ds] = sn(Z[k]['se'], m1)
            if m0.sum() >= 1:
                Bd[ds] = sn(Z[k]['se'], m0)
        p, lo, hi, _ = boot(lambda ks: pooled(A, ks) - pooled(Bd, ks))
        W("  %-5s nn_train [%.1f,%.1f): %+.4f [%+.4f,%+.4f] %s (n_cliff=%d, n_non=%d)"
          % (model, lo_, hi_, p, lo, hi, C.ci_status(lo, hi), n1, n0))
        rows.append(dict(analysis='proximity', model=model, stratum="%.1f-%.1f" % (lo_, hi_),
                         n_cliff=n1, n_non=n0, estimate=p, ci_lo=lo, ci_hi=hi,
                         ci_status=C.ci_status(lo, hi)))

# ================================================================ 7. null predictor
W("")
W("=" * 118)
W("A-3c  CHEMISTRY-FREE NULL PREDICTOR (per-dataset training mean), arm S0")
W("=" * 118)
for d in ['D1', 'D2', 'D0']:
    f = gap_fn('S0', 'SVM', d, source='null')
    p, lo, hi, _ = boot(f)
    W("  train-mean predictor, %s penalty = %+.4f 95%% CI [%+.4f, %+.4f]  %s" % (d, p, lo, hi, C.ci_status(lo, hi)))
    rows.append(dict(analysis='null_predictor', definition=d, estimate=p, ci_lo=lo, ci_hi=hi,
                     ci_status=C.ci_status(lo, hi)))

# ================================================================ 8. model ranking
W("")
W("=" * 118)
W("A-1f  MODEL RANKING STABILITY (per-dataset cliff-RMSE rank, mean over datasets)")
W("=" * 118)
ranks = {}
for arm in ARMS:
    defn = 'D2' if arm == 'R2' else 'D0'
    per = []
    for ds in DATASETS:
        v = []
        for m in MODELS:
            k = (ds, arm, m)
            lab = Z[k]['lab_' + defn] == 1
            v.append(np.sqrt(Z[k]['se'][lab].mean()) if lab.sum() >= 5 else np.nan)
        per.append(st.rankdata(v))
    ranks[arm] = np.nanmean(np.array(per), axis=0)
    W("  %s (%s): " % (arm, defn) + "  ".join("%s=%.2f" % (m, ranks[arm][i]) for i, m in enumerate(MODELS)))
W("  Spearman(mean rank) S0 vs R2 = %.3f ; R0 vs R2 = %.3f"
  % (st.spearmanr(ranks['S0'], ranks['R2']).statistic, st.spearmanr(ranks['R0'], ranks['R2']).statistic))
rows.append(dict(analysis='ranking', estimate=st.spearmanr(ranks['S0'], ranks['R2']).statistic,
                 contrast='spearman_S0_vs_R2'))

pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "A1_analysis_results.csv"), index=False)
open(os.path.join(C.OUT, "A1_analysis.txt"), "w").write("\n".join(LOG))
print("\nwrote A1_analysis.txt and A1_analysis_results.csv")
