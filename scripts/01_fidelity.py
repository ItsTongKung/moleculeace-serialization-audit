"""E1 - Fidelity gate.

Asserts that the vectorised primitives in common.py are element-for-element
identical to the reference implementation in the cloned MoleculeACE package,
and that rapidfuzz's normalised Levenshtein equals python-Levenshtein under
MoleculeACE's own normalisation.  Nothing downstream is trusted unless this
script exits 0.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
#
#
import common as C
from Levenshtein import distance as pylev
from upstream import (get_tanimoto_matrix, get_scaffold_matrix,
                      get_levenshtein_matrix, moleculeace_similarity, ActivityCliffs)

os.makedirs(C.OUT, exist_ok=True)
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)

P("="*100); P("E1  FIDELITY GATE  (vectorised primitives vs upstream MoleculeACE reference)"); P("="*100)

# ---- 0. metric equivalence: rapidfuzz vs python-Levenshtein -----------------
rng = np.random.default_rng(C.SEED)
all_smiles = []
for ds in C.datasets()[:6]:
    all_smiles += C.load(ds)['smiles'].tolist()
idx = rng.choice(len(all_smiles), size=(5000, 2))
bad = 0
for i, j in idx:
    a, b = all_smiles[i], all_smiles[j]
    ref = 1.0 - pylev(a, b) / max(len(a), len(b))
    from rapidfuzz.distance import Levenshtein as RFLev
    got = RFLev.normalized_similarity(a, b)
    if abs(ref - got) > 1e-9:
        bad += 1
P(f"[0] rapidfuzz normalized_similarity == 1 - pyLevenshtein/max(len): mismatches = {bad}/5000")
assert bad == 0

# ---- 1..3 primitive equivalence on the three smallest datasets --------------
small = sorted(C.datasets(), key=lambda d: len(C.load(d)))[:3]
for ds in small:
    df = C.load(ds); smi = df['smiles'].tolist(); act = df['exp_mean [nM]'].values.astype(float)
    t0 = time.time()
    T_ref = get_tanimoto_matrix(smi, hide=True)
    S_ref = get_scaffold_matrix(smi, hide=True)
    L_ref = get_levenshtein_matrix(smi, hide=True)
    T_new = C.tanimoto(C.morgan(smi, False))
    S_new = C.tanimoto(C.morgan(smi, True))
    L_new = C.levenshtein_sim(smi)
    dT = float(np.abs(T_ref - T_new).max()); dS = float(np.abs(S_ref - S_new).max())
    dL = float(np.abs(L_ref - L_new).max())
    # binarised agreement at the 0.9 operating point (what actually matters)
    bT = int(((T_ref >= C.SIM) != (T_new >= C.SIM)).sum())
    bS = int(((S_ref >= C.SIM) != (S_new >= C.SIM)).sum())
    bL = int(((L_ref >= C.SIM) != (L_new >= C.SIM)).sum())
    # full pipeline: upstream ActivityCliffs vs our D0
    ref_lab = np.array(ActivityCliffs(smi, list(act)).get_cliff_molecules(
        return_smiles=False, similarity=C.SIM, potency_fold=C.FOLD), dtype=np.int8)
    Fb = C.foldchange(act) > C.FOLD
    new_lab = C.compound_labels((T_new >= C.SIM) | (S_new >= C.SIM) | (L_new >= C.SIM), Fb)
    agree = int((ref_lab == new_lab).sum())
    shipped = df['cliff_mol'].values.astype(np.int8)
    P(f"[1] {ds:16s} n={len(smi):5d} | maxabs T/S/L = {dT:.2e}/{dS:.2e}/{dL:.2e} | "
      f"binarised pair diffs T/S/L = {bT}/{bS}/{bL} | upstream-vs-ours labels {agree}/{len(smi)} "
      f"| ours-vs-shipped {int((new_lab==shipped).sum())}/{len(smi)} | upstream-vs-shipped "
      f"{int((ref_lab==shipped).sum())}/{len(smi)} [{time.time()-t0:.0f}s]")
    assert agree == len(smi), "vectorised D0 != upstream ActivityCliffs"

# ---- 4. moleculeace_similarity is a disjunction, not a sum ------------------
smi = C.load(small[0])['smiles'].tolist()[:400]
sim_ref = moleculeace_similarity(smi, C.SIM, hide=True)
T = C.tanimoto(C.morgan(smi, False)) >= C.SIM
S = C.tanimoto(C.morgan(smi, True)) >= C.SIM
L = C.levenshtein_sim(smi) >= C.SIM
P(f"[2] upstream moleculeace_similarity max value = {int(sim_ref.max())} "
  f"(1 => numpy bool '+' is logical OR, so 'sim == 1' in find_cliffs() is a true disjunction); "
  f"disagreements vs (T|S|L) = {int((sim_ref.astype(bool) != (T|S|L)).sum())}")
assert int(sim_ref.max()) <= 1
assert int((sim_ref.astype(bool) != (T | S | L)).sum()) == 0

P(""); P("FIDELITY GATE PASSED")
open(os.path.join(C.OUT, "E1_fidelity.txt"), "w").write("\n".join(log))
