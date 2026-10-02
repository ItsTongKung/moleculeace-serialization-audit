"""A-2  Serialization-invariance experiments, with EXPLICIT, DOCUMENTED SEEDS.

Every randomized replicate is generated with an explicit seed passed to the
RDKit random-SMILES generator, so the run is reproducible.

Three rewrite families, in increasing order of adversarialness:
  ROUND-TRIP   RDKit canonical -> parse -> RDKit canonical.  Negative control:
               the benchmark must reproduce under its own shipped convention.
  KEKULE       aromatic-canonical -> Kekule-canonical.  Two equally standard,
               equally valid canonical outputs of the same toolkit.  This is
               the practical convention-sensitivity test.
  RANDOMISED   Chem.MolToRandomSmilesVect(mol, 1, randomSeed=s) for the ten
               explicit seeds 42..51.  Adversarial stress test / mechanism demo.

Every generated string is parsed back and its canonical identity compared with
the original molecule; any mismatch aborts the run.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

CACHE = os.path.join(C.OUT, "cache")
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def jac(a, b):
    inter = int(np.logical_and(a == 1, b == 1).sum())
    uni = int(np.logical_or(a == 1, b == 1).sum())
    return inter / uni if uni else np.nan


rows, perseed = [], []
identity_failures = []
# ---- RESTARTABILITY: resume from partial CSVs written after every dataset ----
_rp = os.path.join(C.OUT, "A2_invariance.csv")
_pp = os.path.join(C.OUT, "A2_invariance_perseed.csv")
_done = set()
if os.path.exists(_rp):
    _d = pd.read_csv(_rp)
    rows = _d.to_dict('records')
    _done = set(_d.dataset)
    if os.path.exists(_pp):
        perseed = pd.read_csv(_pp).to_dict('records')
    print("resuming: %d datasets already done" % len(_done), flush=True)
for k, ds in enumerate(C.datasets()):
    if ds in _done:
        print("[%2d/30] %-18s cached" % (k + 1, ds), flush=True)
        continue
    t0 = time.time()
    df = C.load(ds)
    smi = df['smiles'].tolist()
    n = len(smi)
    az = np.load(os.path.join(CACHE, "adj_" + ds + ".npz"))
    Tb = np.unpackbits(az['packed_Tb'], axis=1)[:, :n].astype(bool)
    Sb = np.unpackbits(az['packed_Sb'], axis=1)[:, :n].astype(bool)
    Fb = np.unpackbits(az['packed_Fb'], axis=1)[:, :n].astype(bool)
    Lb = np.unpackbits(az['packed_Lb'], axis=1)[:, :n].astype(bool)
    base = C.compound_labels(Tb | Sb | Lb, Fb)
    mols = [Chem.MolFromSmiles(s) for s in smi]
    ref_can = [Chem.MolToSmiles(m) for m in mols]
    ref_key = [MolToInchiKey(m) for m in mols]

    def relabel(strings, tag, seed=None):
        bad = 0
        for i, s in enumerate(strings):
            m2 = Chem.MolFromSmiles(s)
            if m2 is None or Chem.MolToSmiles(m2) != ref_can[i]:
                bad += 1
                identity_failures.append((ds, tag, seed, i, smi[i], s))
        assert bad == 0, "IDENTITY FAILURE %s/%s/seed=%s: %d molecules changed" % (ds, tag, seed, bad)
        Lb2 = C.levenshtein_sim(strings) >= C.SIM
        lab = C.compound_labels(Tb | Sb | Lb2, Fb)
        Tb2 = C.tanimoto(C.morgan(strings, False)) >= C.SIM
        Sb2 = C.tanimoto(C.morgan(strings, True)) >= C.SIM
        return lab, int((Tb2 != Tb).sum()), int((Sb2 != Sb).sum())

    rt = ref_can
    lab_rt, cT_rt, cS_rt = relabel(rt, "roundtrip")
    n_rt_str = int(sum(a != b for a, b in zip(smi, rt)))

    kek = []
    for m in mols:
        mk = Chem.Mol(m)
        Chem.Kekulize(mk, clearAromaticFlags=True)
        kek.append(Chem.MolToSmiles(mk, kekuleSmiles=True))
    lab_kk, cT_kk, cS_kk = relabel(kek, "kekule")
    n_kk_str = int(sum(a != b for a, b in zip(smi, kek)))
    kk_key_ok = int(sum(1 for i, s in enumerate(kek)
                        if MolToInchiKey(Chem.MolFromSmiles(s)) == ref_key[i]))

    flips, jacs, ncl = [], [], []
    ever = np.zeros(n, bool)
    for sd in SEEDS:
        rs = [Chem.MolToRandomSmilesVect(m, 1, randomSeed=sd)[0] for m in mols]
        lab, cT, cS = relabel(rs, "random", sd)
        d = (lab != base)
        flips.append(100 * d.mean())
        ever |= d
        jacs.append(jac(lab, base))
        ncl.append(int(lab.sum()))
        perseed.append(dict(dataset=ds, seed=sd, flip_pct=100 * d.mean(),
                            n_cliff_base=int(base.sum()), n_cliff_rand=int(lab.sum()),
                            jaccard=jacs[-1], ctrl_ecfp_pair_diffs=cT,
                            ctrl_scaffold_pair_diffs=cS,
                            n_strings_changed=int(sum(a != b for a, b in zip(smi, rs)))))

    rows.append(dict(dataset=ds, n=n, n_cliff_base=int(base.sum()),
                     rt_strings_changed_pct=100 * n_rt_str / n,
                     rt_flip_pct=100 * float((lab_rt != base).mean()),
                     rt_n_cliff=int(lab_rt.sum()), rt_jaccard=jac(lab_rt, base),
                     kek_strings_changed_pct=100 * n_kk_str / n,
                     kek_flip_pct=100 * float((lab_kk != base).mean()),
                     kek_n_cliff=int(lab_kk.sum()), kek_jaccard=jac(lab_kk, base),
                     kek_gain=int((lab_kk > base).sum()), kek_loss=int((lab_kk < base).sum()),
                     kek_inchikey_identical="%d/%d" % (kk_key_ok, n),
                     rand_flip_mean_pct=float(np.mean(flips)),
                     rand_flip_sd_pct=float(np.std(flips, ddof=1)),
                     rand_flip_min_pct=float(np.min(flips)),
                     rand_flip_max_pct=float(np.max(flips)),
                     rand_jaccard_mean=float(np.mean(jacs)),
                     rand_n_cliff_mean=float(np.mean(ncl)),
                     ever_flip_pct=100 * float(ever.mean()),
                     ctrl_ecfp_pair_diffs=cT_rt + cT_kk,
                     ctrl_scaffold_pair_diffs=cS_rt + cS_kk,
                     secs=round(time.time() - t0, 1)))
    r = rows[-1]
    print("[%2d/30] %-18s n=%5d base=%4d | roundtrip %5.2f%% | kekule %5.2f%% (J=%.3f, %.0f%% strings differ) "
          "| random %5.2f+-%.2f%% (J=%.3f) | ctrlT/S=%d/%d [%.0fs]"
          % (k + 1, ds, n, r['n_cliff_base'], r['rt_flip_pct'], r['kek_flip_pct'], r['kek_jaccard'],
             r['kek_strings_changed_pct'], r['rand_flip_mean_pct'], r['rand_flip_sd_pct'],
             r['rand_jaccard_mean'], r['ctrl_ecfp_pair_diffs'], r['ctrl_scaffold_pair_diffs'],
             r['secs']), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "A2_invariance.csv"), index=False)
    pd.DataFrame(perseed).to_csv(os.path.join(C.OUT, "A2_invariance_perseed.csv"), index=False)

o = pd.DataFrame(rows)
ps = pd.DataFrame(perseed)
o.to_csv(os.path.join(C.OUT, "A2_invariance.csv"), index=False)
ps.to_csv(os.path.join(C.OUT, "A2_invariance_perseed.csv"), index=False)
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


P("=" * 100)
P("A-2  SERIALIZATION INVARIANCE  (explicit seeds " + ",".join(map(str, SEEDS)) + ")")
P("=" * 100)
P("identity verification: %d failures across all rewrites and seeds (must be 0)" % len(identity_failures))
P("POSITIVE CONTROLS: ECFP4 pair diffs total=%d, generic-scaffold pair diffs total=%d (must be 0)"
  % (int(o.ctrl_ecfp_pair_diffs.sum()) + int(ps.ctrl_ecfp_pair_diffs.sum()),
     int(o.ctrl_scaffold_pair_diffs.sum()) + int(ps.ctrl_scaffold_pair_diffs.sum())))
P("")
P("ROUND-TRIP  strings changed median %.2f%%  ->  labels flipped median %.3f%% (max %.3f%%)  Jaccard median %.4f"
  % (o.rt_strings_changed_pct.median(), o.rt_flip_pct.median(), o.rt_flip_pct.max(), o.rt_jaccard.median()))
P("KEKULE      strings changed median %.2f%%  ->  labels flipped median %.2f%% (range %.2f-%.2f%%)  "
  "Jaccard median %.3f; total cliffs %d -> %d (%+.1f%%)"
  % (o.kek_strings_changed_pct.median(), o.kek_flip_pct.median(), o.kek_flip_pct.min(),
     o.kek_flip_pct.max(), o.kek_jaccard.median(), int(o.n_cliff_base.sum()), int(o.kek_n_cliff.sum()),
     100 * (o.kek_n_cliff.sum() - o.n_cliff_base.sum()) / o.n_cliff_base.sum()))
P("RANDOMISED  labels flipped median %.2f%% (range %.2f-%.2f%%)  compound-weighted mean %.2f%%  "
  "Jaccard median %.3f (range %.3f-%.3f)"
  % (o.rand_flip_mean_pct.median(), o.rand_flip_min_pct.min(), o.rand_flip_max_pct.max(),
     (o.rand_flip_mean_pct * o.n).sum() / o.n.sum(), o.rand_jaccard_mean.median(),
     o.rand_jaccard_mean.min(), o.rand_jaccard_mean.max()))
P("            per-dataset SD across the ten seeds: median %.3f pp, max %.3f pp; flipped in >=1 seed median %.2f%%"
  % (o.rand_flip_sd_pct.median(), o.rand_flip_sd_pct.max(), o.ever_flip_pct.median()))
open(os.path.join(C.OUT, "A2_invariance_summary.txt"), "w").write("\n".join(L))
if identity_failures:
    pd.DataFrame(identity_failures, columns=['dataset', 'rewrite', 'seed', 'idx', 'orig', 'new']).to_csv(
        os.path.join(C.OUT, "A2_identity_failures.csv"), index=False)
