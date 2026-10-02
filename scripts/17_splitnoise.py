"""E4  MAGNITUDE-MATCHED SPLIT-REALISATION CONTROL.

The manuscript's drift control (R0 - S0) perturbs the split in only 12 of 30
datasets, by a mean of 4.28% of compounds, whereas the primary contrast
(R2 - R0) carries a median 31.3% membership change in all 30.  The control is
therefore a lower bound and cannot answer the question: could split-realisation variability of COMPARABLE MAGNITUDE move the
cliff penalty as much as the R2 - R0 shift?

Design (fixed before the run; no post-hoc seed selection):
  * cliff definition held FIXED at D0 -- the labels never change, so nothing
    about the benchmark's answer key is perturbed;
  * NREP = 24 independent split realisations per dataset, generated with the
    SAME algorithm as every other arm (spectral clustering of the ECFP4
    Tanimoto matrix into 5 clusters, then a per-cluster 80/20 train_test_split
    stratified on the D0 label), varying ONLY the random_state, which is set to
    the pre-declared arithmetic sequence 1000, 1001, ..., 1023 for both the
    clustering and the stratified draw;
  * the four scikit-learn families retrained from scratch on each realisation
    with the repository's published per-dataset hyperparameters, ECFP4-1024
    features, exactly as in 05_fullrepair.py;
  * the penalty for each realisation is the own-complement D0 penalty pooled
    over the 30 datasets, computed with the same estimator as Table 1.

Reported:
  * the distribution of the pooled D0 penalty over the 24 realisations
    (mean, SD, min, max, and the 2.5-97.5 percentile span);
  * membership perturbation of each realisation against the R0 reference split,
    so the control can be CALIBRATED against the ~31% carried by the primary
    repair;
  * the distribution of pairwise differences between realisations, which is the
    quantity directly comparable with the R2 - R0 point estimate: if a
    same-definition split re-draw of matched magnitude routinely moves the
    penalty by as much as R2 - R0 did, the primary contrast cannot be
    attributed to the definition.

Restartable: per (dataset, rep, model) prediction files are skipped if present.
Outputs: OUT/E9_splitnoise_perrep.csv, OUT/E9_splitnoise.csv, OUT/E9_splitnoise.txt
"""
import os, sys, time, yaml, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')

CACHE = os.path.join(C.OUT, "cache")
PRED = os.path.join(C.PRED, "preds_splitnoise")
os.makedirs(PRED, exist_ok=True)
NREP = 24
STATES = list(range(1000, 1000 + NREP))          # pre-declared, not tuned
NJOBS = 2
# Model panel for this control, and why one family.  The 24 pre-declared realisations are kept
# unchanged -- that is the design parameter the control depends on -- and the MODEL panel is
# narrowed instead.  Measured per-fit costs on the larger datasets (n ~ 2,750) are: SVM 5.5 s,
# spectral clustering 7.7 s, and RF 215 s, because the repository's published RF configuration is
# n_estimators=1000.  RF alone would therefore have cost ~40x the rest of the experiment combined.
# SVM is retained because every headline figure in the paper uses it; RF and GBM are dropped on
# cost, and KNN because its numbers are not interpretable at all under the distance-tie finding of
# 19_knnties.py.  All exclusions were decided on measured cost and on the pre-existing KNN result,
# BEFORE any split-noise distribution was inspected.
MODELS = {'SVM': lambda h: SVR(**h)}


def split_at(Tm, labels, state, n):
    """MoleculeACE's split algorithm with random_state replaced by `state`
    in BOTH the spectral clustering and the per-cluster stratified draw."""
    sc = SpectralClustering(n_clusters=C.NCLUST, random_state=state,
                            affinity='precomputed')
    clusters = sc.fit(Tm).labels_
    tr, te = [], []
    for c in range(C.NCLUST):
        ci = np.where(clusters == c)[0]
        if ci.size == 0:
            continue
        cc = [int(labels[i]) for i in ci]
        if sum(cc) > 2 and ci.size >= 5:
            a, b = train_test_split(ci, test_size=C.TESTSIZE, random_state=state,
                                    stratify=cc, shuffle=True)
        else:
            a, b = train_test_split(ci, test_size=C.TESTSIZE, random_state=state,
                                    shuffle=True)
        tr.extend(a); te.extend(b)
    sp = np.array(['train'] * n, dtype=object)
    sp[np.array(te, int)] = 'test'
    return sp


rows = []
t_start = time.time()
for k, ds in enumerate(C.datasets()):
    t0 = time.time()
    z = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
    Tm = np.load(os.path.join(CACHE, "T_%s.npy" % ds))
    X = np.unpackbits(z['X'], axis=1)[:, :1024]
    y = z['y']
    lab0 = z['lab_D0']
    ref = z['split_reconD0'].astype(str)          # R0 reference split
    n = len(y)
    for rep, state in enumerate(STATES):
        spf = os.path.join(PRED, "%s__rep%02d__split.npy" % (ds, rep))
        if os.path.exists(spf):
            sp = np.load(spf, allow_pickle=True).astype(str)
        else:
            sp = split_at(Tm, lab0, state, n)
            np.save(spf, sp.astype(object), allow_pickle=True)
        tr, te = sp == 'train', sp == 'test'
        memb_change = 100.0 * float((sp != ref).mean())
        test_frac = float(te.mean())
        for mname, ctor in MODELS.items():
            fout = os.path.join(PRED, "%s__rep%02d__%s_ECFP.npz" % (ds, rep, mname))
            cfgp = os.path.join(C.CFG, ds, "%s_ECFP.yml" % mname)
            if not os.path.exists(cfgp):
                continue
            if os.path.exists(fout):
                zz = np.load(fout, allow_pickle=True)
                yh, yt, lb = zz['y_pred'], zz['y_true'], zz['lab_D0']
            else:
                h = yaml.safe_load(open(cfgp)) or {}
                h = {a: b for a, b in h.items() if a != 'epochs'}
                mdl = ctor(h); mdl.fit(X[tr], y[tr]); yh = mdl.predict(X[te])
                yt, lb = y[te], lab0[te]
                np.savez_compressed(fout, y_true=yt, y_pred=yh, lab_D0=lb,
                                    dataset=ds, rep=rep, random_state=state, model=mname)
            se = (yt - yh) ** 2
            m = lb == 1
            rows.append(dict(dataset=ds, rep=rep, random_state=state, model=mname,
                             n_test=int(te.sum()), test_frac=test_frac,
                             memb_change_vs_R0=memb_change,
                             n_cliff=int(m.sum()),
                             ss_cliff=float(se[m].sum()), n_c=int(m.sum()),
                             ss_non=float(se[~m].sum()), n_n=int((~m).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "E9_splitnoise_perrep.csv"), index=False)
    print("[%2d/30] %-18s %d reps x %d models  membership change vs R0: mean %.2f%%  [%.0fs]"
          % (k + 1, ds, NREP, len(MODELS),
             np.mean([r['memb_change_vs_R0'] for r in rows if r['dataset'] == ds]),
             time.time() - t0), flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(C.OUT, "E9_splitnoise_perrep.csv"), index=False)

LOG = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

W("=" * 122)
W("E4  MAGNITUDE-MATCHED SPLIT-REALISATION CONTROL  (D0 labels FIXED; %d split realisations, random_state %d..%d)"
  % (NREP, STATES[0], STATES[-1]))
W("=" * 122)

# ---- calibration of the perturbation magnitude
mc = R.groupby(['dataset', 'rep']).memb_change_vs_R0.first()
per_ds = mc.groupby('dataset').mean()
W("PERTURBATION MAGNITUDE (calibration against the primary repair)")
W("  membership change vs the R0 reference split: mean %.2f%%, median %.2f%%, range %.2f-%.2f%% over %d realisations x 30 datasets"
  % (mc.mean(), mc.median(), mc.min(), mc.max(), NREP))
W("  per-dataset mean: median %.2f%%, range %.2f-%.2f%%" % (per_ds.median(), per_ds.min(), per_ds.max()))
tf = R.groupby(['dataset', 'rep']).test_frac.first()
ceil = (2 * tf * (1 - tf) * 100)
W("  independent-redraw ceiling 2p(1-p) using each realisation's own test fraction: mean %.2f%%" % ceil.mean())
W("  PRIMARY REPAIR carries a median 31.3% membership change (D2 vs R0 split) -- see E3_cache_summary.csv")
W("  => this control is magnitude-MATCHED if the figure above is close to 31%; the manuscript's")
W("     R0-S0 control was 4.28% and therefore was not.")
W("")

# ---- pooled penalty per realisation
W("POOLED D0 CLIFF PENALTY BY REALISATION  (own-complement, pooled over 30 datasets, same estimator as Table 1)")
hdr = "%-5s %9s %9s %9s %9s %9s %11s" % ("model", "mean", "SD", "min", "max", "p2.5-97.5", "R0 ref")
W(hdr); W("-" * len(hdr))
srows = []
for m in sorted(set(R.model)):
    sub = R[R.model == m]
    vals = []
    for rep in range(NREP):
        s = sub[sub.rep == rep]
        if s.empty:
            continue
        gap = (np.sqrt(s.ss_cliff.sum() / s.n_c.sum()) -
               np.sqrt(s.ss_non.sum() / s.n_n.sum()))
        vals.append(gap)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    W("%-5s %+9.4f %9.4f %+9.4f %+9.4f  %+.4f/%+.4f" % (m, vals.mean(), vals.std(ddof=1),
                                                        vals.min(), vals.max(), lo, hi))
    # pairwise differences between realisations = the split-noise-only analogue of R2-R0
    dif = vals[:, None] - vals[None, :]
    iu = np.triu_indices(len(vals), 1)
    dv = dif[iu]
    srows.append(dict(model=m, n_reps=len(vals), penalty_mean=vals.mean(),
                      penalty_sd=vals.std(ddof=1), penalty_min=vals.min(),
                      penalty_max=vals.max(), penalty_p2_5=lo, penalty_p97_5=hi,
                      pairwise_abs_mean=np.abs(dv).mean(),
                      pairwise_abs_median=float(np.median(np.abs(dv))),
                      pairwise_abs_p95=float(np.percentile(np.abs(dv), 95)),
                      pairwise_range=vals.max() - vals.min()))
S = pd.DataFrame(srows)
W("")
W("SPLIT-NOISE-ONLY ANALOGUE OF THE PRIMARY CONTRAST")
W("  |difference| in pooled penalty between two same-definition split realisations:")
W("  %-5s %12s %12s %12s %12s" % ("model", "mean|d|", "median|d|", "p95|d|", "full range"))
for _, r in S.iterrows():
    W("  %-5s %12.4f %12.4f %12.4f %12.4f"
      % (r.model, r.pairwise_abs_mean, r.pairwise_abs_median, r.pairwise_abs_p95, r.pairwise_range))
W("")
W("CALIBRATION AGAINST THE OBSERVED PRIMARY CONTRAST (R2-R0 from A1_analysis_results.csv):")
try:
    an = pd.read_csv(os.path.join(C.OUT, "A1_analysis_results.csv"))
    pc = an[(an.analysis == 'contrast') & (an.contrast.astype(str).str.contains('PRIMARY'))]
    for _, r in S.iterrows():
        row = pc[pc.model == r.model]
        if row.empty:
            continue
        obs = float(row.estimate.iloc[0])
        frac = float((np.abs(np.array([obs])) <= r.pairwise_abs_p95).mean())
        W("  %-5s observed R2-R0 = %+.4f ; split-noise p95|d| = %.4f  -> observed effect is %s"
          % (r.model, obs, r.pairwise_abs_p95,
             "WITHIN the split-noise band (split realisation of matched magnitude could produce it)"
             if abs(obs) <= r.pairwise_abs_p95 else
             "OUTSIDE the split-noise band"))
        S.loc[S.model == r.model, 'observed_R2_minus_R0'] = obs
        S.loc[S.model == r.model, 'within_splitnoise_p95'] = bool(abs(obs) <= r.pairwise_abs_p95)
except Exception as e:
    W("  (could not read A1_analysis_results.csv: %s)" % e)

S.to_csv(os.path.join(C.OUT, "E9_splitnoise.csv"), index=False)
W("")
W("total wall time %.1f min" % ((time.time() - t_start) / 60))
W("")
W("INTERPRETATION (fixed before the run): if the observed R2-R0 shift lies inside the distribution of")
W("differences produced by same-definition split re-draws of comparable membership perturbation, then")
W("split-realisation variability alone can account for a shift of the observed size, and the primary")
W("contrast must not be attributed to the cliff definition.  This is reported whichever way it falls.")
open(os.path.join(C.OUT, "E9_splitnoise.txt"), "w").write("\n".join(LOG))
print("\nwrote E9_splitnoise.{csv,txt}, E9_splitnoise_perrep.csv")
