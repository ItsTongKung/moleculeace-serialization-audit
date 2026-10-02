"""F  KNN DISTANCE-TIE SENSITIVITY  (reproducibility diagnosis).

Independent re-execution of the full-repair pipeline in a freshly reconstructed
environment at the same pinned library versions reproduced the SVM and GBM arms
BIT-IDENTICALLY (0 of 90 dataset x arm cells differed) and the RF arm to within
5.4e-4 RMSE, but reproduced NO cell of the KNN arm exactly (90 of 90 differed;
maximum |dRMSE| 0.0201, maximum |dpenalty| 0.0672).

Mechanism, measured here.  MoleculeACE's published KNN hyperparameters are
`metric: euclidean`, `n_neighbors: 5`, `weights: distance`, applied to 1024-bit
BINARY ECFP4 vectors.  Squared Euclidean distance between two binary vectors is
the integer Hamming distance, so the distance spectrum is coarse and exact ties
are pervasive.  When the k-th and (k+1)-th neighbours are exactly equidistant,
which one enters the neighbourhood is not determined by the hyperparameters:
scikit-learn's neighbour selection is not order-stable across search backends
(and, for the brute-force path, not across BLAS/SIMD code paths), so two
machines running identical versions can return different predictions.

This script measures, for every dataset and arm:
  * the fraction of test compounds with an exact distance tie at the k-th
    neighbour boundary (the exposed population);
  * the pooled D0/D2 cliff penalty and overall RMSE under the three neighbour
    search backends scikit-learn exposes (brute, ball_tree, kd_tree), which
    bounds the tie-induced uncertainty of every KNN number in the paper.

Nothing here changes the estimator used in the primary analysis: the published
hyperparameters are retained unchanged for faithfulness.  The spread reported
here is the honest uncertainty attached to the KNN column.

Outputs: OUT/E11_knnties.csv, OUT/E11_knnties_pooled.csv, OUT/E11_knnties.txt
"""
import os, sys, time, yaml, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
warnings.filterwarnings('ignore')

CACHE = os.path.join(C.OUT, "cache")
ARMS = {'S0': ('shipped_split', 'D0'), 'R0': ('split_reconD0', 'D0'),
        'R2': ('split_reconD2', 'D2')}
BACKENDS = ['brute', 'ball_tree', 'kd_tree']

rows = []
_rp = os.path.join(C.OUT, "E11_knnties.csv")
_done = set()
if os.path.exists(_rp):
    _d = pd.read_csv(_rp); rows = _d.to_dict('records'); _done = set(_d.dataset)
    print("resuming: %d datasets done" % len(_done), flush=True)

for k, ds in enumerate(C.datasets()):
    if ds in _done:
        print("[%2d/30] %-18s cached" % (k + 1, ds), flush=True); continue
    t0 = time.time()
    z = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
    # dtype EXACTLY as the primary pipeline produces it (uint8 from unpackbits);
    # a float64 copy is compared separately below because the dtype turns out to be
    # a second, independent lever on which neighbour wins a tie.
    Xu = np.unpackbits(z['X'], axis=1)[:, :1024]
    X = Xu
    y = z['y']
    cfgp = os.path.join(C.CFG, ds, "KNN_ECFP.yml")
    h = yaml.safe_load(open(cfgp)) or {}
    h = {a: b for a, b in h.items() if a != 'epochs'}
    kk = int(h.get('n_neighbors', 5))
    for arm, (spkey, defn) in ARMS.items():
        sp = z[spkey].astype(str)
        tr, te = sp == 'train', sp == 'test'
        lab = z['lab_' + defn][te] == 1
        # --- tie exposure at the k-th neighbour boundary
        nn = NearestNeighbors(n_neighbors=min(kk + 1, int(tr.sum())),
                              metric='euclidean', algorithm='brute').fit(X[tr])
        dist, _ = nn.kneighbors(X[te])
        if dist.shape[1] > kk:
            tie = np.abs(dist[:, kk - 1] - dist[:, kk]) < 1e-9
        else:
            tie = np.zeros(dist.shape[0], bool)
        r = dict(dataset=ds, arm=arm, definition=defn, n_test=int(te.sum()),
                 n_cliff=int(lab.sum()), k=kk,
                 tie_at_k_pct=100.0 * float(tie.mean()),
                 n_distinct_distances=int(np.unique(np.round(dist, 9)).size))
        # --- penalty and RMSE under each backend
        preds = {}
        for be in BACKENDS:
            try:
                m = KNeighborsRegressor(n_jobs=2, algorithm=be, **h)
                m.fit(X[tr], y[tr]); yh = m.predict(X[te])
            except Exception as e:
                r['err_' + be] = str(e)[:60]; continue
            preds[be] = yh
            se = (y[te] - yh) ** 2
            r['rmse_' + be] = float(np.sqrt(se.mean()))
            r['ss_cliff_' + be] = float(se[lab].sum()); r['n_c'] = int(lab.sum())
            r['ss_non_' + be] = float(se[~lab].sum()); r['n_n'] = int((~lab).sum())
            r['gap_' + be] = (float(np.sqrt(se[lab].mean())) -
                              float(np.sqrt(se[~lab].mean()))) if lab.sum() >= 5 else np.nan
        # --- dtype lever: same backend, uint8 vs float64 features
        try:
            m8 = KNeighborsRegressor(n_jobs=2, algorithm='brute', **h)
            m8.fit(Xu[tr], y[tr]); p8 = m8.predict(Xu[te])
            m64 = KNeighborsRegressor(n_jobs=2, algorithm='brute', **h)
            m64.fit(Xu[tr].astype(np.float64), y[tr]); p64 = m64.predict(Xu[te].astype(np.float64))
            r['dtype_max_pred_diff'] = float(np.abs(p8 - p64).max())
            r['dtype_n_pred_diff'] = int((np.abs(p8 - p64) > 1e-12).sum())
            se8 = (y[te] - p8) ** 2; se64 = (y[te] - p64) ** 2
            r['gap_uint8'] = (float(np.sqrt(se8[lab].mean())) - float(np.sqrt(se8[~lab].mean()))) if lab.sum() >= 5 else np.nan
            r['gap_float64'] = (float(np.sqrt(se64[lab].mean())) - float(np.sqrt(se64[~lab].mean()))) if lab.sum() >= 5 else np.nan
            r['ss_cliff_uint8'] = float(se8[lab].sum()); r['ss_non_uint8'] = float(se8[~lab].sum())
            r['ss_cliff_float64'] = float(se64[lab].sum()); r['ss_non_float64'] = float(se64[~lab].sum())
        except Exception as e:
            r['dtype_err'] = str(e)[:60]
        if len(preds) >= 2:
            ks = list(preds)
            r['max_pred_diff_between_backends'] = float(max(
                np.abs(preds[a] - preds[b]).max() for a in ks for b in ks if a < b))
            r['n_pred_diff_brute_vs_ball'] = int(
                (np.abs(preds['brute'] - preds['ball_tree']) > 1e-12).sum()
            ) if 'ball_tree' in preds and 'brute' in preds else -1
        rows.append(r)
    pd.DataFrame(rows).to_csv(_rp, index=False)
    print("[%2d/30] %-18s ties@k S0/R0/R2 = %.1f/%.1f/%.1f%%  maxpreddiff %.3f  [%.0fs]"
          % (k + 1, ds,
             *[next(x['tie_at_k_pct'] for x in rows if x['dataset'] == ds and x['arm'] == a)
               for a in ['S0', 'R0', 'R2']],
             max(x.get('max_pred_diff_between_backends', 0) for x in rows if x['dataset'] == ds),
             time.time() - t0), flush=True)

R = pd.DataFrame(rows)
R.to_csv(_rp, index=False)

LOG = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

W("=" * 118)
W("F  KNN DISTANCE-TIE SENSITIVITY")
W("=" * 118)
W("Published KNN hyperparameters: metric=euclidean, weights=distance, k per dataset (median k=%d)."
  % int(R.k.median()))
W("Features: 1024-bit BINARY ECFP4, so squared Euclidean distance is the integer Hamming distance.")
W("")
W("TIE EXPOSURE (test compounds whose k-th and (k+1)-th neighbours are exactly equidistant):")
for arm in ['S0', 'R0', 'R2']:
    s = R[R.arm == arm]
    W("  arm %s: mean %.1f%%, median %.1f%%, range %.1f-%.1f%% of test compounds"
      % (arm, s.tie_at_k_pct.mean(), s.tie_at_k_pct.median(),
         s.tie_at_k_pct.min(), s.tie_at_k_pct.max()))
W("  pooled over all arms: mean %.1f%% of test compounds sit on a tie boundary" % R.tie_at_k_pct.mean())
W("")
W("PREDICTION DIVERGENCE BETWEEN NEIGHBOUR-SEARCH BACKENDS (same versions, same data):")
W("  maximum |prediction difference| over all dataset x arm cells: %.4f log units"
  % R.max_pred_diff_between_backends.max())
W("  mean over cells of the per-cell maximum: %.4f" % R.max_pred_diff_between_backends.mean())
W("  cells in which brute and ball_tree disagree on >=1 compound: %d of %d"
  % (int((R.n_pred_diff_brute_vs_ball > 0).sum()), len(R)))

W("")
W("FEATURE-DTYPE LEVER (identical backend 'brute', identical hyperparameters; uint8 vs float64 features):")
W("  cells where the two dtypes give different predictions: %d of %d"
  % (int((R.dtype_n_pred_diff > 0).sum()), len(R)))
W("  maximum |prediction difference| attributable to dtype alone: %.4f log units"
  % R.dtype_max_pred_diff.max())
for arm in ['S0', 'R0', 'R2']:
    s_ = R[R.arm == arm]
    g8 = (np.sqrt(s_.ss_cliff_uint8.sum() / s_.n_c.sum()) - np.sqrt(s_.ss_non_uint8.sum() / s_.n_n.sum()))
    g64 = (np.sqrt(s_.ss_cliff_float64.sum() / s_.n_c.sum()) - np.sqrt(s_.ss_non_float64.sum() / s_.n_n.sum()))
    W("  arm %s pooled penalty: uint8 %+.4f  float64 %+.4f  (difference %+.4f)" % (arm, g8, g64, g64 - g8))
_a = {}
for arm in ['R0', 'R2']:
    s_ = R[R.arm == arm]
    _a[(arm, 'u')] = (np.sqrt(s_.ss_cliff_uint8.sum() / s_.n_c.sum()) - np.sqrt(s_.ss_non_uint8.sum() / s_.n_n.sum()))
    _a[(arm, 'f')] = (np.sqrt(s_.ss_cliff_float64.sum() / s_.n_c.sum()) - np.sqrt(s_.ss_non_float64.sum() / s_.n_n.sum()))
W("  PRIMARY CONTRAST R2-R0: uint8 %+.4f   float64 %+.4f"
  % (_a[('R2', 'u')] - _a[('R0', 'u')], _a[('R2', 'f')] - _a[('R0', 'f')]))

prows = []
W("")
W("POOLED CLIFF PENALTY AND OVERALL RMSE BY BACKEND (own-complement, pooled over 30 datasets)")
W("  %-4s %-11s %12s %12s %12s %10s" % ("arm", "quantity", "brute", "ball_tree", "kd_tree", "spread"))
for arm in ['S0', 'R0', 'R2']:
    s = R[R.arm == arm]
    vals = {}
    for be in BACKENDS:
        if ('ss_cliff_' + be) not in s:
            continue
        gap = (np.sqrt(s['ss_cliff_' + be].sum() / s.n_c.sum()) -
               np.sqrt(s['ss_non_' + be].sum() / s.n_n.sum()))
        vals[be] = gap
    if vals:
        sp = max(vals.values()) - min(vals.values())
        W("  %-4s %-11s %12.4f %12.4f %12.4f %10.4f"
          % (arm, "penalty", vals.get('brute', np.nan), vals.get('ball_tree', np.nan),
             vals.get('kd_tree', np.nan), sp))
        prows.append(dict(arm=arm, quantity='penalty', **{('gap_' + b): vals.get(b) for b in BACKENDS},
                          spread=sp))
    rv = {}
    for be in BACKENDS:
        if ('rmse_' + be) in s:
            # pooled RMSE over datasets from the per-dataset sums of squares
            ss = (s['ss_cliff_' + be] + s['ss_non_' + be]).sum()
            nn_ = (s.n_c + s.n_n).sum()
            rv[be] = float(np.sqrt(ss / nn_))
    if rv:
        sp = max(rv.values()) - min(rv.values())
        W("  %-4s %-11s %12.4f %12.4f %12.4f %10.4f"
          % (arm, "RMSE_all", rv.get('brute', np.nan), rv.get('ball_tree', np.nan),
             rv.get('kd_tree', np.nan), sp))
        prows.append(dict(arm=arm, quantity='rmse_all', **{('gap_' + b): rv.get(b) for b in BACKENDS},
                          spread=sp))
P = pd.DataFrame(prows)
P.to_csv(os.path.join(C.OUT, "E11_knnties_pooled.csv"), index=False)

# primary contrast under each backend
W("")
W("PRIMARY CONTRAST R2 - R0 UNDER EACH BACKEND (the quantity whose sign the paper declines to claim):")
for be in BACKENDS:
    if ('ss_cliff_' + be) not in R:
        continue
    g = {}
    for arm in ['R0', 'R2']:
        s = R[R.arm == arm]
        g[arm] = (np.sqrt(s['ss_cliff_' + be].sum() / s.n_c.sum()) -
                  np.sqrt(s['ss_non_' + be].sum() / s.n_n.sum()))
    W("  %-10s R2-R0 = %+.4f" % (be, g['R2'] - g['R0']))
W("")
W("CONCLUSION FOR THE MANUSCRIPT")
W("  The KNN column is not bit-reproducible across machines at fixed library versions, because the")
W("  published hyperparameters do not determine the neighbourhood when distances tie, and ties are")
W("  pervasive on binary fingerprints under a Euclidean metric. The spread above is the honest")
W("  uncertainty on every KNN figure; it is of the same order as the primary contrast itself, so no")
W("  KNN-specific direction may be claimed. SVM and GBM reproduced bit-identically and RF to within")
W("  5.4e-4 RMSE, so this is a KNN-specific limitation, not a general failure of the pipeline.")
W("  It is also an instance of the paper's own thesis: a benchmark contract that omits an")
W("  implementation detail leaves the reported number under-determined.")
open(os.path.join(C.OUT, "E11_knnties.txt"), "w").write("\n".join(LOG))
print("\nwrote E11_knnties.{csv,txt}, E11_knnties_pooled.csv")
