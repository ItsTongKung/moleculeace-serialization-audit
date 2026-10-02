"""Writes the released per-compound resources from the analysis cache.

  labels/<dataset>_invariant_labels.csv     cliff labels under every definition (D0-D3)
  splits/<dataset>_splits.csv               shipped, reconstructed-D0 (R0) and repaired-D2 (R2)
                                            splits, cluster identities, nearest-training-neighbour
                                            ECFP4 similarity
  splits/realizations/<dataset>_split_realizations.csv
                                            the 24 same-definition split realizations of the
                                            magnitude-matched control (17_splitnoise.py)

Inputs: OUT/cache (03_cache.py, 20_d3capped.py), PRED/preds_splitnoise (17_splitnoise.py).
Outputs go to OUT/resources/; reproduce.py compares them with the released copies.
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

CACHE = os.path.join(C.OUT, "cache")
DEST = os.path.join(C.OUT, "resources")
for d in ("labels", "splits", os.path.join("splits", "realizations")):
    os.makedirs(os.path.join(DEST, d), exist_ok=True)

for ds in C.datasets():
    z = np.load(os.path.join(CACHE, ds + ".npz"), allow_pickle=True)
    raw = C.load(ds)
    assert list(raw["smiles"]) == list(z["smiles"]), ds
    lab = pd.DataFrame({
        "smiles": z["smiles"], "y_pEC50_pKi": z["y"], "exp_mean_nM": raw["exp_mean [nM]"].values,
        "cliff_mol_shipped": z["shipped_cliff"], "D0_recomputed": z["lab_D0"],
        "D1_ecfp4_strict": z["lab_D1"], "D2_invariant": z["lab_D2"], "D3_mmp": z["lab_D3"]})
    capf = os.path.join(CACHE, f"labD3capped_{ds}.npy")
    if os.path.exists(capf):
        lab["D3_mmp_size_capped_13"] = np.load(capf)
    lab.to_csv(os.path.join(DEST, "labels", ds + "_invariant_labels.csv"), index=False)
    sp = pd.DataFrame({
        "smiles": z["smiles"], "split_shipped": z["shipped_split"],
        "split_R0_from_D0": z["split_reconD0"], "split_R2_from_D2": z["split_reconD2"],
        "cluster_D0": z["clusters_D0"], "cluster_D2": z["clusters_D2"],
        "nn_ecfp4_sim_to_train_shipped": z["nn_shipped"],
        "nn_ecfp4_sim_to_train_R0": z["nn_reconD0"], "nn_ecfp4_sim_to_train_R2": z["nn_reconD2"]})
    sp.to_csv(os.path.join(DEST, "splits", ds + "_splits.csv"), index=False)
    reps = sorted(glob.glob(os.path.join(C.PRED, "preds_splitnoise", f"{ds}__rep*__split.npy")))
    if reps:
        cols = {}
        for i, f in enumerate(reps):
            cols["rep%02d_state%d" % (i, 1000 + i)] = np.load(f, allow_pickle=True)
        pd.DataFrame(cols).to_csv(os.path.join(DEST, "splits", "realizations",
                                               ds + "_split_realizations.csv"), index=False)
    print(f"{ds:18s} exported", flush=True)
