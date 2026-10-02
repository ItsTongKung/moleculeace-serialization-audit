"""Shared primitives for the MoleculeACE serialization audit.

Every function here is a vectorised re-implementation of a MoleculeACE primitive.
Equivalence to the upstream reference implementation is asserted by
`scripts/01_fidelity.py`, which runs the actual upstream `ActivityCliffs` /
`split_data` code on sample datasets and compares element-by-element.

Cliff-definition vocabulary (fixed for the whole study):
  D0  ECFP4>=0.9  OR anonymised-graph>=0.9 OR normalised-Levenshtein>=0.9   (as shipped)
  D1  ECFP4>=0.9                                                            (invariant)
  D2  ECFP4>=0.9  OR anonymised-graph>=0.9                                  (invariant)
  D3  matched molecular pair, single cut, shared canonical core             (invariant)
A pair is a cliff pair iff the similarity criterion holds AND fold-change > 10.
A compound is cliff-labelled iff it participates in >=1 cliff pair.
"""
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rapidfuzz.distance import Levenshtein as RFLev
from rapidfuzz import process as rfprocess
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split

RDLogger.DisableLog('rdApp.*')

# Paths.  Every location can be overridden with an environment variable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.environ.get("MOLECULEACE_DIR", os.path.join(ROOT, "external", "MoleculeACE"))
DATA = os.path.join(UPSTREAM, "MoleculeACE", "Data", "benchmark_data")
CFG  = os.path.join(UPSTREAM, "MoleculeACE", "Data", "configures", "benchmark")
OUT  = os.environ.get("MSA_OUT", os.path.join(ROOT, "outputs"))          # analysis tables + cache
PRED = os.environ.get("MSA_PRED", OUT)                                   # per-molecule predictions
FIGDIR = os.environ.get("MSA_FIGDIR", os.path.join(OUT, "figures"))      # rendered figures
FIGDATA = os.environ.get("MSA_FIGDATA", os.path.join(OUT, "figure_source_data"))
for _d in (OUT, FIGDIR, FIGDATA):
    os.makedirs(_d, exist_ok=True)
SIM, FOLD, SEED, NCLUST, TESTSIZE = 0.9, 10, 42, 5, 0.2


def datasets():
    return sorted(f[:-4] for f in os.listdir(DATA)
                  if f.startswith("CHEMBL") and f.endswith(".csv"))


def load(ds):
    return pd.read_csv(os.path.join(DATA, ds + ".csv"))


# ---------------------------------------------------------------- similarity
def morgan(smiles, generic=False):
    """1024-bit ECFP4, optionally on the element- and bond-order-anonymised whole molecular
    graph (RDKit MakeScaffoldGeneric, which retains side chains and is NOT a Bemis-Murcko
    framework).  Falls back to GetScaffoldForMol on failure, exactly as upstream
    cliffs.get_scaffold_matrix() does."""
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if generic:
            try:
                m = GraphFramework(m)
            except Exception:
                m = GetScaffoldForMol(m)
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024))
    return out


def tanimoto(fps):
    n = len(fps); M = np.zeros((n, n), np.float32)
    for i in range(n):
        M[i, i:] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i:])
    M = np.maximum(M, M.T); np.fill_diagonal(M, 0.0)
    return M


def levenshtein_sim(smiles):
    """1 - d(a,b)/max(|a|,|b|), exactly MoleculeACE's normalisation."""
    M = rfprocess.cdist(smiles, smiles, scorer=RFLev.normalized_similarity,
                        dtype=np.float32, workers=-1)
    np.fill_diagonal(M, 0.0)
    return M


def foldchange(activity_nM):
    a = np.asarray(activity_nM, float)
    M = np.maximum.outer(a, a) / np.minimum.outer(a, a)
    np.fill_diagonal(M, 0.0)
    return M


def mmp_pairs(smiles):
    """Single-cut MMP: two molecules are a pair iff they share a canonical core."""
    n = len(smiles); idx = {}
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        try:
            cuts = FragmentMol(m, maxCuts=1, resultsAsMols=False)
        except Exception:
            cuts = []
        keys = set()
        for cut in cuts:
            for part in [c for c in cut if c]:
                fr = part.split('.')
                if len(fr) < 2:
                    continue
                a, b = Chem.MolFromSmiles(fr[0]), Chem.MolFromSmiles(fr[-1])
                if a is None or b is None:
                    continue
                core = fr[0] if a.GetNumAtoms() >= b.GetNumAtoms() else fr[-1]
                keys.add(Chem.MolToSmiles(Chem.MolFromSmiles(core)))
        for k in keys:
            idx.setdefault(k, []).append(i)
    M = np.zeros((n, n), bool)
    for k, v in idx.items():
        if len(v) > 1:
            v = np.array(v); M[np.ix_(v, v)] = True
    np.fill_diagonal(M, False)
    return M


def compound_labels(pair_ok, fold_ok):
    """Compound is a cliff compound iff it is in >=1 admitted pair with >10-fold."""
    return np.logical_and(pair_ok, fold_ok).any(axis=0).astype(np.int8)


# ---------------------------------------------------------------- split
def moleculeace_split(tanimoto_matrix, cliff_labels, n=None):
    """Byte-for-byte the algorithm of MoleculeACE.benchmark.data_prep.split_data:
    spectral clustering (precomputed Tanimoto affinity, 5 clusters, seed 42),
    then a per-cluster 80/20 train_test_split stratified on the cliff label
    (unstratified when a cluster holds <=2 cliff compounds)."""
    n = len(cliff_labels) if n is None else n
    spectral = SpectralClustering(n_clusters=NCLUST, random_state=SEED,
                                 affinity='precomputed')
    clusters = spectral.fit(tanimoto_matrix).labels_
    train_idx, test_idx = [], []
    for c in range(NCLUST):
        ci = np.where(clusters == c)[0]
        cc = [int(cliff_labels[i]) for i in ci]
        if sum(cc) > 2:
            a, b = train_test_split(ci, test_size=TESTSIZE, random_state=SEED,
                                    stratify=cc, shuffle=True)
        else:
            a, b = train_test_split(ci, test_size=TESTSIZE, random_state=SEED,
                                    shuffle=True)
        train_idx.extend(a); test_idx.extend(b)
    split = np.array(['train'] * n, dtype=object)
    split[np.array(test_idx, int)] = 'test'
    return split, clusters


# ---------------------------------------------------------------- statistics
def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def pooled_rmse(per_dataset_sq_err, keys):
    e = [per_dataset_sq_err[k] for k in keys if k in per_dataset_sq_err]
    if not e:
        return np.nan
    return float(np.sqrt(np.concatenate(e).mean()))


def cluster_bootstrap(stat_fn, dataset_keys, B=4000, seed=SEED):
    """95% CI by resampling DATASETS with replacement (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    point = stat_fn(list(dataset_keys))
    bs = []
    keys = list(dataset_keys)
    for _ in range(B):
        s = list(rng.choice(keys, size=len(keys), replace=True))
        v = stat_fn(s)
        if v is not None and np.isfinite(v):
            bs.append(v)
    if len(bs) < 100:
        return point, np.nan, np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi)


def ci_status(lo, hi):
    """Correct two-sided verdict. A wholly negative interval also excludes zero."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undetermined"
    return "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
