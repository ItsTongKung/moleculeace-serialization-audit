"""E2 - Can the shipped (corrected) MoleculeACE train/test split be reproduced?

The split is: SpectralClustering(5, precomputed ECFP4-Tanimoto affinity, seed 42)
-> per-cluster 80/20 train_test_split stratified on the shipped cliff label.
Everything except the spectral eigendecomposition is bit-deterministic.  We
therefore scan the solver options that SpectralClustering exposes and report,
per dataset, the best achievable agreement with the shipped `split` column.
This quantifies software drift instead of hiding it.
"""
import os, sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split

os.makedirs(C.OUT, exist_ok=True)
os.makedirs(os.path.join(C.OUT, "cache"), exist_ok=True)

VARIANTS = {
    "default_arpack": dict(eigen_solver=None, assign_labels="kmeans", n_init=10),
    "lobpcg":         dict(eigen_solver="lobpcg", assign_labels="kmeans", n_init=10),
    "kmeans_n_init1": dict(eigen_solver=None, assign_labels="kmeans", n_init=1),
    "discretize":     dict(eigen_solver=None, assign_labels="discretize", n_init=10),
}

def split_from_clusters(clusters, cliff, n):
    tr, te = [], []
    for c in sorted(set(clusters)):
        ci = np.where(clusters == c)[0]
        cc = [int(cliff[i]) for i in ci]
        if sum(cc) > 2:
            a, b = train_test_split(ci, test_size=C.TESTSIZE, random_state=C.SEED,
                                    stratify=cc, shuffle=True)
        else:
            a, b = train_test_split(ci, test_size=C.TESTSIZE, random_state=C.SEED, shuffle=True)
        tr.extend(a); te.extend(b)
    s = np.array(['train'] * n, dtype=object); s[np.array(te, int)] = 'test'
    return s

rows = []
for k, ds in enumerate(C.datasets()):
    t0 = time.time()
    df = C.load(ds); smi = df['smiles'].tolist()
    shipped = df['split'].values; cliff = df['cliff_mol'].values.astype(int)
    T = C.tanimoto(C.morgan(smi, False))
    np.save(os.path.join(C.OUT, "cache", f"T_{ds}.npy"), T)
    r = dict(dataset=ds, n=len(smi), n_test_shipped=int((shipped == 'test').sum()))
    for name, kw in VARIANTS.items():
        try:
            cl = SpectralClustering(n_clusters=C.NCLUST, random_state=C.SEED,
                                    affinity='precomputed', **kw).fit(T).labels_
            s = split_from_clusters(cl, cliff, len(smi))
            r[f"agree_{name}"] = 100.0 * float((s == shipped).mean())
        except Exception as e:
            r[f"agree_{name}"] = np.nan
            r[f"err_{name}"] = str(e)[:60]
    rows.append(r)
    best = max((r[f"agree_{n}"] for n in VARIANTS if np.isfinite(r.get(f"agree_{n}", np.nan))), default=np.nan)
    print(f"[{k+1:2d}/30] {ds:18s} n={len(smi):5d} " +
          " ".join(f"{n}={r[f'agree_{n}']:6.2f}%" for n in VARIANTS) +
          f" | best={best:6.2f}% [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "E2_split_reproduction.csv"), index=False)

d = pd.DataFrame(rows)
print("\n=== SPLIT REPRODUCTION SUMMARY ===")
for n in VARIANTS:
    c = d[f"agree_{n}"]
    print(f"  {n:16s} median {c.median():6.2f}%  min {c.min():6.2f}%  n_exact(=100%) {int((c>99.999).sum())}/30")
best = d[[f"agree_{n}" for n in VARIANTS]].max(axis=1)
print(f"  best-of-variants  median {best.median():6.2f}%  min {best.min():6.2f}%  n_exact {int((best>99.999).sum())}/30")
