"""E4 - Cache the cliff-PAIR adjacency (similarity criterion AND >10-fold) for
each definition, packed to bits.  Needed for partner-location analysis under
every split realisation, and for the branch operating-region analysis."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

CACHE = os.path.join(C.OUT, "cache")
for k, ds in enumerate(C.datasets()):
    out = os.path.join(CACHE, f"adj_{ds}.npz")
    if os.path.exists(out):
        print(f"[{k+1:2d}/30] {ds:18s} cached", flush=True); continue
    t0 = time.time()
    df = C.load(ds); smi = df['smiles'].tolist()
    act = df['exp_mean [nM]'].values.astype(float); n = len(smi)
    Tm = np.load(os.path.join(CACHE, f"T_{ds}.npy"))
    Sb = C.tanimoto(C.morgan(smi, True)) >= C.SIM
    Lb = C.levenshtein_sim(smi) >= C.SIM
    Tb = Tm >= C.SIM
    Fb = C.foldchange(act) > C.FOLD
    MM = C.mmp_pairs(smi)
    adj = {'D0': (Tb | Sb | Lb) & Fb, 'D1': Tb & Fb,
           'D2': (Tb | Sb) & Fb, 'D3': MM & Fb}
    np.savez_compressed(out, n=n, **{f"adj_{a}": np.packbits(b, axis=1) for a, b in adj.items()},
                        packed_Tb=np.packbits(Tb, axis=1), packed_Sb=np.packbits(Sb, axis=1),
                        packed_Lb=np.packbits(Lb, axis=1), packed_Fb=np.packbits(Fb, axis=1))
    print(f"[{k+1:2d}/30] {ds:18s} n={n:5d} pairs D0={int(adj['D0'].sum())//2:6d} "
          f"D1={int(adj['D1'].sum())//2:5d} D2={int(adj['D2'].sum())//2:6d} D3={int(adj['D3'].sum())//2:7d} "
          f"[{time.time()-t0:.0f}s]", flush=True)
print("ADJACENCY DONE")
