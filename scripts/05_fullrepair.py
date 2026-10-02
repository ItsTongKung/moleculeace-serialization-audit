"""A-1  FULL-REPAIR COUNTERFACTUAL.

Three arms, all with the repository's published per-dataset hyperparameters,
ECFP4-1024 features, seed 42, and NO new hyperparameter tuning:

  S0  shipped split          -> train -> evaluate under D0   (benchmark as distributed)
  R0  split reconstructed from D0 labels -> train -> evaluate under D0
  R2  split reconstructed from D2 labels -> RETRAIN -> evaluate under D2

R2 - R0 is the primary, causally clean estimand: same code, same seed, same
software stack, same hyperparameters; the ONLY thing that differs is the cliff
definition used to label and to stratify the split.
R0 - S0 is an internal drift control: same cliff definition, different split
realisation, so it measures how much split realisation alone moves the penalty.
S0 -> R2 is the bridge comparison and is reported separately.

Restartable: per (dataset, arm, model) prediction files are skipped if present.
"""
import os, sys, time, yaml, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')

CACHE = os.path.join(C.OUT, "cache")
PRED = os.path.join(C.PRED, "preds_fullrepair"); os.makedirs(PRED, exist_ok=True)
NJOBS = 3
MODELS = {'SVM': lambda h: SVR(**h),
          'RF':  lambda h: RandomForestRegressor(random_state=C.SEED, n_jobs=NJOBS, **h),
          'GBM': lambda h: GradientBoostingRegressor(random_state=C.SEED, **h),
          'KNN': lambda h: KNeighborsRegressor(n_jobs=NJOBS, **h)}
# arm -> (split key in cache, cliff definition that defines the benchmark in that arm)
ARMS = {'S0': ('shipped_split',  'D0'),
        'R0': ('split_reconD0',  'D0'),
        'R2': ('split_reconD2',  'D2')}

rows = []
for k, ds in enumerate(C.datasets()):
    t0 = time.time()
    z = np.load(os.path.join(CACHE, f"{ds}.npz"), allow_pickle=True)
    az = np.load(os.path.join(CACHE, f"adj_{ds}.npz"))
    n = int(az['n'])
    X = np.unpackbits(z['X'], axis=1)[:, :1024]
    y = z['y']; smi = z['smiles']
    Tm = np.load(os.path.join(CACHE, f"T_{ds}.npy"))
    lab = {d: z[f'lab_{d}'] for d in ['D0', 'D1', 'D2', 'D3']}
    adj = {d: np.unpackbits(az[f'adj_{d}'], axis=1)[:, :n].astype(bool)
           for d in ['D0', 'D1', 'D2', 'D3']}
    for arm, (spkey, defn) in ARMS.items():
        sp = z[spkey].astype(str)
        tr, te = sp == 'train', sp == 'test'
        nn_train = Tm[:, tr].max(axis=1).astype(np.float32)
        # partner in training set, per definition
        partner_tr = {d: adj[d][:, tr].any(axis=1) for d in adj}
        partner_te = {d: adj[d][:, te].any(axis=1) for d in adj}
        for mname, ctor in MODELS.items():
            fout = os.path.join(PRED, f"{ds}__{arm}__{mname}_ECFP.npz")
            cfgp = os.path.join(C.CFG, ds, f"{mname}_ECFP.yml")
            if not os.path.exists(cfgp):
                continue
            if os.path.exists(fout):
                zz = np.load(fout, allow_pickle=True); yh = zz['y_pred']; secs = float(zz['secs'])
            else:
                h = yaml.safe_load(open(cfgp)) or {}
                h = {a: b for a, b in h.items() if a != 'epochs'}
                t1 = time.time()
                mdl = ctor(h); mdl.fit(X[tr], y[tr]); yh = mdl.predict(X[te])
                secs = time.time() - t1
                np.savez_compressed(fout, y_true=y[te], y_pred=yh, smiles=smi[te],
                                    arm=arm, definition=defn, model=mname, dataset=ds,
                                    nn_train=nn_train[te], secs=secs,
                                    **{f'lab_{d}': lab[d][te] for d in lab},
                                    **{f'partner_train_{d}': partner_tr[d][te] for d in adj},
                                    **{f'partner_test_{d}': partner_te[d][te] for d in adj})
            r = dict(dataset=ds, arm=arm, definition=defn, model=mname,
                     n_train=int(tr.sum()), n_test=int(te.sum()),
                     rmse_all=C.rmse(y[te], yh), secs=round(secs, 1))
            for d in ['D0', 'D1', 'D2', 'D3']:
                m = lab[d][te] == 1
                r[f'n_{d}'] = int(m.sum())
                r[f'rmse_{d}'] = C.rmse(y[te][m], yh[m]) if m.sum() >= 5 else np.nan
                r[f'rmse_non{d}'] = C.rmse(y[te][~m], yh[~m]) if (~m).sum() >= 5 else np.nan
                r[f'gap_{d}'] = r[f'rmse_{d}'] - r[f'rmse_non{d}']
            r['gap_own'] = r[f'gap_{defn}']          # the arm's own headline penalty
            r['rmse_cliff_own'] = r[f'rmse_{defn}']
            r['n_cliff_own'] = r[f'n_{defn}']
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "A1_fullrepair_results.csv"), index=False)
    got = {a: [x for x in rows if x['dataset'] == ds and x['arm'] == a and x['model'] == 'SVM'] for a in ARMS}
    msg = " ".join(f"{a}:gap={got[a][0]['gap_own']:+.3f}(n={got[a][0]['n_cliff_own']:4d})"
                   for a in ARMS if got[a])
    print(f"[{k+1:2d}/30] {ds:18s} SVM {msg} [{time.time()-t0:.0f}s]", flush=True)
print("FULL-REPAIR TRAINING DONE")
