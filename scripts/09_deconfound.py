"""A-3d  De-confounding arms: MACCS keys (167 bits) and the repository's eleven
physicochemical descriptors, trained under all three split arms.

Purpose: the D1/D2 cliff definitions are built from ECFP4 similarity while the
primary models use ECFP4 features, so a near-identical fingerprint with a very
different potency is irreducibly hard for an ECFP4 model.  MACCS and PHYSCHEM
models do not share that representation.  PHYSCHEM carries no substructure
information at all and is the conservative estimate.

Descriptors are computed exactly as in MoleculeACE.benchmark.featurization:
MACCS via rdkit MACCSkeys.GenMACCSKeys; PHYSCHEM as the eleven properties of
compute_physchem(), standardised on the training fold only.
"""
import os, sys, time, yaml, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors, Crippen, QED, rdMolDescriptors, rdmolops, DataStructs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')

CACHE = os.path.join(C.OUT, "cache")
PRED = os.path.join(C.PRED, "preds_alt"); os.makedirs(PRED, exist_ok=True)
NJOBS = 3
MODELS = {'SVM': lambda h: SVR(**h),
          'RF':  lambda h: RandomForestRegressor(random_state=C.SEED, n_jobs=NJOBS, **h),
          'GBM': lambda h: GradientBoostingRegressor(random_state=C.SEED, **h),
          'KNN': lambda h: KNeighborsRegressor(n_jobs=NJOBS, **h)}
ARMS = {'S0': ('shipped_split', 'D0'), 'R0': ('split_reconD0', 'D0'), 'R2': ('split_reconD2', 'D2')}


def maccs(smiles):
    X = np.zeros((len(smiles), 167), np.uint8)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)), X[i])
    return X


def physchem(smiles):
    X = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        X.append([Descriptors.ExactMolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m),
                  Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
                  m.GetNumAtoms(), m.GetNumHeavyAtoms(), Crippen.MolMR(m),
                  QED.properties(m).PSA, rdmolops.GetFormalCharge(m),
                  rdMolDescriptors.CalcNumRings(m)])
    return np.array(X, float)


rows = []
for k, ds in enumerate(C.datasets()):
    t0 = time.time()
    z = np.load(os.path.join(CACHE, ds + ".npz"), allow_pickle=True)
    smi = list(z['smiles']); y = z['y']
    lab = {d: z['lab_' + d] for d in ['D0', 'D1', 'D2', 'D3']}
    F = {'MACCS': maccs(smi), 'PHYSCHEM': physchem(smi)}
    for arm, (spkey, defn) in ARMS.items():
        sp = z[spkey].astype(str); tr, te = sp == 'train', sp == 'test'
        for desc, Xfull in F.items():
            if desc == 'PHYSCHEM':
                sc = StandardScaler().fit(Xfull[tr])
                X = sc.transform(Xfull)
            else:
                X = Xfull
            for mname, ctor in MODELS.items():
                fout = os.path.join(PRED, "%s__%s__%s_%s.npz" % (ds, arm, mname, desc))
                cfgp = os.path.join(C.CFG, ds, "%s_%s.yml" % (mname, desc))
                if not os.path.exists(cfgp):
                    continue
                if os.path.exists(fout):
                    yh = np.load(fout, allow_pickle=True)['y_pred']
                else:
                    h = yaml.safe_load(open(cfgp)) or {}
                    h = {a: b for a, b in h.items() if a != 'epochs'}
                    mdl = ctor(h); mdl.fit(X[tr], y[tr]); yh = mdl.predict(X[te])
                    np.savez_compressed(fout, y_true=y[te], y_pred=yh,
                                        smiles=np.array(smi, dtype=object)[te],
                                        arm=arm, definition=defn, model=mname, descriptor=desc,
                                        dataset=ds, **{'lab_' + d: lab[d][te] for d in lab})
                r = dict(dataset=ds, arm=arm, descriptor=desc, model=mname,
                         rmse_all=C.rmse(y[te], yh))
                for d in ['D0', 'D1', 'D2', 'D3']:
                    m = lab[d][te] == 1
                    r['n_' + d] = int(m.sum())
                    r['rmse_' + d] = C.rmse(y[te][m], yh[m]) if m.sum() >= 5 else np.nan
                    r['gap_' + d] = (r['rmse_' + d] - C.rmse(y[te][~m], yh[~m])) if m.sum() >= 5 else np.nan
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "A5_deconfound_results.csv"), index=False)
    print("[%2d/30] %-18s done [%.0fs]" % (k + 1, ds, time.time() - t0), flush=True)
print("DECONFOUND TRAINING DONE")
