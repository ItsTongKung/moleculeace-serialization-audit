"""E3 - Per-dataset cache: labels under D0/D1/D2/D3, three split realisations,
ECFP4 feature matrix, and the raw similarity values needed by the
operating-region analysis.  Restartable: skips datasets already cached.

Three split realisations are produced, all with the SAME code, seed and
software stack:
  shipped   the `split` column as distributed by MoleculeACE
  recon_D0  our reconstruction, stratified on D0 labels   (isolates drift)
  recon_D2  our reconstruction, stratified on D2 labels   (the repaired benchmark)
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem
from rdkit.Chem import DataStructs

CACHE = os.path.join(C.OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
rows = []
for k, ds in enumerate(C.datasets()):
    fp_out = os.path.join(CACHE, f"{ds}.npz")
    t0 = time.time()
    df = C.load(ds)
    smi = df['smiles'].tolist()
    act = df['exp_mean [nM]'].values.astype(float)
    y = df['y [pEC50/pKi]'].values.astype(float)
    shipped_split = df['split'].values.astype(str)
    shipped_cliff = df['cliff_mol'].values.astype(np.int8)
    n = len(smi)

    if os.path.exists(fp_out):
        z = np.load(fp_out, allow_pickle=True)
        rows.append({kk: z[kk].item() if z[kk].shape == () else None
                     for kk in ['summary']}['summary'])
        print(f"[{k+1:2d}/30] {ds:18s} cached", flush=True)
        continue

    Tm = np.load(os.path.join(CACHE, f"T_{ds}.npy"))          # from 02
    fps_g = C.morgan(smi, True)
    Sm = C.tanimoto(fps_g)
    Lm = C.levenshtein_sim(smi)
    Fb = C.foldchange(act) > C.FOLD
    Tb, Sb, Lb = Tm >= C.SIM, Sm >= C.SIM, Lm >= C.SIM
    MM = C.mmp_pairs(smi)

    lab = {'D0': C.compound_labels(Tb | Sb | Lb, Fb),
           'D1': C.compound_labels(Tb, Fb),
           'D2': C.compound_labels(Tb | Sb, Fb),
           'D3': C.compound_labels(MM, Fb)}

    # ---- split realisations (identical algorithm, identical seed) ----------
    sp0, cl0 = C.moleculeace_split(Tm, lab['D0'], n)
    sp2, cl2 = C.moleculeace_split(Tm, lab['D2'], n)

    # ---- ECFP4 feature matrix --------------------------------------------
    fps_e = C.morgan(smi, False)
    X = np.zeros((n, 1024), np.uint8)
    for i, v in enumerate(fps_e):
        DataStructs.ConvertToNumpyArray(v, X[i])

    # ---- operating-region data: ECFP4 similarity of cliff pairs by branch --
    iu = np.triu_indices(n, 1)
    cliffpair = Fb[iu] & (Tb | Sb | Lb)[iu]
    t_i, s_i, l_i = Tb[iu], Sb[iu], Lb[iu]
    tv = Tm[iu].astype(np.float32)
    branch = {}
    for nm, mask in [('ecfp_admitted',  cliffpair & t_i),
                     ('scaf_admitted',  cliffpair & s_i),
                     ('str_admitted',   cliffpair & l_i),
                     ('ecfp_only',      cliffpair & t_i & ~s_i & ~l_i),
                     ('scaf_only',      cliffpair & s_i & ~t_i & ~l_i),
                     ('str_only',       cliffpair & l_i & ~t_i & ~s_i),
                     ('all_cliffpairs', cliffpair)]:
        branch[f"tsim_{nm}"] = tv[mask]

    # ---- nearest-neighbour ECFP4 similarity to each split's training set ---
    nn = {}
    for nmsp, sp in [('shipped', shipped_split), ('reconD0', sp0), ('reconD2', sp2)]:
        tr = (sp == 'train')
        nn[f"nn_{nmsp}"] = Tm[:, tr].max(axis=1).astype(np.float32)

    summary = dict(dataset=ds, n=n,
                   n_cliff_shipped=int(shipped_cliff.sum()),
                   n_cliff_D0=int(lab['D0'].sum()), n_cliff_D1=int(lab['D1'].sum()),
                   n_cliff_D2=int(lab['D2'].sum()), n_cliff_D3=int(lab['D3'].sum()),
                   label_agree_shipped_vs_D0=100.0 * float((lab['D0'] == shipped_cliff).mean()),
                   split_agree_reconD0_vs_shipped=100.0 * float((sp0 == shipped_split).mean()),
                   split_change_D2_vs_reconD0=100.0 * float((sp2 != sp0).mean()),
                   split_change_D2_vs_shipped=100.0 * float((sp2 != shipped_split).mean()),
                   n_cliffpairs_D0=int(cliffpair.sum()),
                   n_pairs_str_only=int((cliffpair & l_i & ~t_i & ~s_i).sum()),
                   n_pairs_scaf_only=int((cliffpair & s_i & ~t_i & ~l_i).sum()),
                   n_pairs_ecfp_only=int((cliffpair & t_i & ~s_i & ~l_i).sum()),
                   n_pairs_ecfp_admitted=int((cliffpair & t_i).sum()),
                   n_pairs_scaf_admitted=int((cliffpair & s_i).sum()),
                   n_pairs_str_admitted=int((cliffpair & l_i).sum()),
                   secs=round(time.time() - t0, 1))

    np.savez_compressed(fp_out, smiles=np.array(smi, dtype=object), y=y, act=act,
                        X=np.packbits(X, axis=1),
                        shipped_split=shipped_split, shipped_cliff=shipped_cliff,
                        split_reconD0=sp0.astype(str), split_reconD2=sp2.astype(str),
                        clusters_D0=cl0, clusters_D2=cl2,
                        **{f"lab_{a}": b for a, b in lab.items()},
                        **branch, **nn, summary=np.array(summary, dtype=object))
    rows.append(summary)
    print(f"[{k+1:2d}/30] {ds:18s} n={n:5d} cliffs D0/D1/D2/D3="
          f"{summary['n_cliff_D0']:4d}/{summary['n_cliff_D1']:4d}/{summary['n_cliff_D2']:4d}/{summary['n_cliff_D3']:4d} "
          f"| labelrepro={summary['label_agree_shipped_vs_D0']:7.3f}% "
          f"| splitrepro={summary['split_agree_reconD0_vs_shipped']:6.2f}% "
          f"| splitChange(D2 vs reconD0)={summary['split_change_D2_vs_reconD0']:5.2f}% "
          f"[{summary['secs']}s]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "E3_cache_summary.csv"), index=False)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(C.OUT, "E3_cache_summary.csv"), index=False)
print("\n=== CACHE SUMMARY ===")
print(f"label reproduction (D0 vs shipped cliff_mol): median {d.label_agree_shipped_vs_D0.median():.3f}%  "
      f"min {d.label_agree_shipped_vs_D0.min():.3f}%  exact {int((d.label_agree_shipped_vs_D0>99.9999).sum())}/30")
print(f"split reproduction (reconD0 vs shipped):      median {d.split_agree_reconD0_vs_shipped.median():.2f}%  "
      f"min {d.split_agree_reconD0_vs_shipped.min():.2f}%  exact {int((d.split_agree_reconD0_vs_shipped>99.9999).sum())}/30")
print(f"split membership change D2 vs reconD0:        median {d.split_change_D2_vs_reconD0.median():.2f}%  "
      f"range {d.split_change_D2_vs_reconD0.min():.2f}-{d.split_change_D2_vs_reconD0.max():.2f}%")
print(f"split membership change D2 vs shipped:        median {d.split_change_D2_vs_shipped.median():.2f}%")
print(f"cliff prevalence: D0 {100*d.n_cliff_D0.sum()/d.n.sum():.1f}%  D1 {100*d.n_cliff_D1.sum()/d.n.sum():.1f}%  "
      f"D2 {100*d.n_cliff_D2.sum()/d.n.sum():.1f}%  D3 {100*d.n_cliff_D3.sum()/d.n.sum():.1f}%")
print(f"cliff PAIRS D0 total {int(d.n_cliffpairs_D0.sum())}; string-only {int(d.n_pairs_str_only.sum())} "
      f"({100*d.n_pairs_str_only.sum()/d.n_cliffpairs_D0.sum():.1f}%); scaffold-only {int(d.n_pairs_scaf_only.sum())}; "
      f"ecfp-only {int(d.n_pairs_ecfp_only.sum())}")
