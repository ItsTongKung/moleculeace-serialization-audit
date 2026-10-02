"""E1  CROSS-SERIALIZATION / CROSS-TOOLKIT PRACTICAL ARM.

The invariance experiments in 06_invariance.py all produce their alternative
SMILES with RDKit.  That establishes non-invariance inside one toolkit
(output-mode change, version drift, adversarial randomisation) but leaves the
practical magnitude of the *cross-toolkit* scenario unmeasured.

This script adds three molecule-preserving serialization routes that do NOT
come from RDKit's own canonical SMILES writer:

  ARM A  inchi_rt   RDKit mol -> InChI -> reconstruct mol -> RDKit canonical SMILES.
                    The InChI layer is an independent, IUPAC-standard
                    canonicalisation of the molecular graph, so the atom
                    ordering that reaches the SMILES writer is decided outside
                    RDKit's own canonical ranking.
  ARM C1 indigo     shipped SMILES -> EPAM Indigo loadMolecule -> canonicalSmiles().
                    Genuinely independent C++ toolkit, independent perception
                    and canonicalisation.
  ARM C2 openbabel  shipped SMILES -> Open Babel (pybel) -> write('can').
                    Second genuinely independent toolkit.

ARM B (a ChEMBL `canonical_smiles` arm) is DECLINED, not skipped silently: the
distributed benchmark files carry only the columns
`smiles, exp_mean [nM], y, cliff_mol, split, y [pEC50/pKi]` and no ChEMBL
molecule identifier (a missing-identifier issue is open upstream as molML/
MoleculeACE#16).  Re-deriving identifiers by structure search would introduce
ambiguous entity matching, which the protocol excludes.

IDENTITY DISCIPLINE (non-negotiable).  For every molecule and every arm the
rewritten string is parsed back with RDKit and both (i) the RDKit canonical
SMILES and (ii) the InChIKey are compared with the original molecule's.  A
molecule that fails either check is QUARANTINED: its shipped string is kept
unchanged for that arm, and it is counted and reported as a toolkit
round-trip failure, never as label instability.  Two label-change estimates are
therefore reported:

  primary      identity-preserved molecules rewritten, quarantined molecules
               left as shipped (this is what a real user importing through the
               foreign toolkit would obtain, minus any molecule the toolkits
               disagree about)
  clean-subset the same statistic restricted to datasets with ZERO quarantined
               molecules, so that no residual identity question can be carried
               into the label-change figure

Outputs: OUT/E6_crosstoolkit.csv, OUT/E6_crosstoolkit_quarantine.csv,
         OUT/E6_crosstoolkit.txt
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, MolFromInchi, MolToInchiKey

CACHE = os.path.join(C.OUT, "cache")

# ------------------------------------------------------------------ toolkits
HAVE_INDIGO = HAVE_OB = False
try:
    from indigo import Indigo
    _ind = Indigo()
    INDIGO_VERSION = _ind.version()
    HAVE_INDIGO = True
except Exception as e:                                           # pragma: no cover
    INDIGO_VERSION = "unavailable (%s)" % e
try:
    from openbabel import pybel
    OB_VERSION = pybel.ob.OBReleaseVersion()
    HAVE_OB = True
except Exception as e:                                           # pragma: no cover
    OB_VERSION = "unavailable (%s)" % e


def w_inchi_rt(mol, shipped):
    """RDKit mol -> standard InChI -> mol -> RDKit canonical SMILES."""
    ik = MolToInchi(mol)
    if not ik:
        return None
    m2 = MolFromInchi(ik)
    if m2 is None:
        return None
    return Chem.MolToSmiles(m2)


def w_indigo(mol, shipped):
    """shipped SMILES -> Indigo -> Indigo canonical SMILES."""
    try:
        m = _ind.loadMolecule(shipped)
        return m.canonicalSmiles()
    except Exception:
        return None


def w_openbabel(mol, shipped):
    """shipped SMILES -> Open Babel -> Open Babel canonical SMILES."""
    try:
        m = pybel.readstring("smi", shipped)
        s = m.write("can").strip().split("\t")[0].split()[0]
        return s or None
    except Exception:
        return None


ARMS = [("inchi_rt", w_inchi_rt, "RDKit InChI round-trip (IUPAC canonicalisation layer)", True)]
if HAVE_INDIGO:
    ARMS.append(("indigo", w_indigo, "EPAM Indigo %s canonical SMILES" % INDIGO_VERSION, True))
if HAVE_OB:
    ARMS.append(("openbabel", w_openbabel, "Open Babel %s canonical SMILES" % OB_VERSION, True))


def jac(a, b):
    inter = int(np.logical_and(a == 1, b == 1).sum())
    uni = int(np.logical_or(a == 1, b == 1).sum())
    return inter / uni if uni else np.nan


rows, quar = [], []
_rp = os.path.join(C.OUT, "E6_crosstoolkit.csv")
_qp = os.path.join(C.OUT, "E6_crosstoolkit_quarantine.csv")
_done = set()
if os.path.exists(_rp):
    _d = pd.read_csv(_rp); rows = _d.to_dict('records')
    _narms = len(ARMS)
    _cnt = _d.groupby('dataset').size()
    _done = set(_cnt[_cnt >= _narms].index)
    rows = [r for r in rows if r['dataset'] in _done]
    if os.path.exists(_qp):
        quar = pd.read_csv(_qp).fillna('').to_dict('records')
        quar = [q for q in quar if q['dataset'] in _done]
    print("resuming: %d datasets already done" % len(_done), flush=True)
for k, ds in enumerate(C.datasets()):
    if ds in _done:
        print("[%2d/30] %-18s cached" % (k + 1, ds), flush=True); continue
    t0 = time.time()
    df = C.load(ds)
    smi = df['smiles'].tolist()
    n = len(smi)
    az = np.load(os.path.join(CACHE, "adj_%s.npz" % ds))
    Tb = np.unpackbits(az['packed_Tb'], axis=1)[:, :n].astype(bool)
    Sb = np.unpackbits(az['packed_Sb'], axis=1)[:, :n].astype(bool)
    Lb = np.unpackbits(az['packed_Lb'], axis=1)[:, :n].astype(bool)
    Fb = np.unpackbits(az['packed_Fb'], axis=1)[:, :n].astype(bool)
    base = C.compound_labels(Tb | Sb | Lb, Fb)

    mols = [Chem.MolFromSmiles(s) for s in smi]
    ref_can = [Chem.MolToSmiles(m) for m in mols]
    ref_key = [MolToInchiKey(m) for m in mols]

    for arm, fn, desc, _ in ARMS:
        out, nq, nfail_write, nfail_can, nfail_key = [], 0, 0, 0, 0
        for i, m in enumerate(mols):
            s2 = fn(m, smi[i])
            ok = False
            if s2 is None:
                nfail_write += 1
            else:
                m2 = Chem.MolFromSmiles(s2)
                if m2 is None:
                    nfail_can += 1
                else:
                    can_ok = Chem.MolToSmiles(m2) == ref_can[i]
                    key_ok = MolToInchiKey(m2) == ref_key[i]
                    if can_ok and key_ok:
                        ok = True
                    elif not can_ok:
                        nfail_can += 1
                    else:
                        nfail_key += 1
            if ok:
                out.append(s2)
            else:
                nq += 1
                out.append(smi[i])          # keep shipped string; never counted as instability
                quar.append(dict(dataset=ds, arm=arm, idx=i, shipped=smi[i],
                                 rewritten=s2 if s2 is not None else "",
                                 reason=("write_failed" if s2 is None else
                                         "canonical_smiles_mismatch" if nfail_can else "inchikey_mismatch")))

        # graph-derived positive controls recomputed from the rewritten strings
        Tb2 = C.tanimoto(C.morgan(out, False)) >= C.SIM
        Sb2 = C.tanimoto(C.morgan(out, True)) >= C.SIM
        ctrlT, ctrlS = int((Tb2 != Tb).sum()), int((Sb2 != Sb).sum())

        Lb2 = C.levenshtein_sim(out) >= C.SIM
        lab = C.compound_labels(Tb | Sb | Lb2, Fb)
        d = (lab != base)
        nchanged = int(sum(a != b for a, b in zip(smi, out)))
        Lp, Lo = np.array([len(s) for s in smi]), np.array([len(s) for s in out])
        rows.append(dict(dataset=ds, arm=arm, n=n,
                         quarantined=nq, quarantined_pct=100.0 * nq / n,
                         q_write_failed=nfail_write, q_can_mismatch=nfail_can, q_key_mismatch=nfail_key,
                         strings_changed_pct=100.0 * nchanged / n,
                         flip_pct=100.0 * float(d.mean()),
                         n_cliff_base=int(base.sum()), n_cliff_new=int(lab.sum()),
                         gain=int((lab > base).sum()), loss=int((lab < base).sum()),
                         jaccard=jac(lab, base),
                         mean_len_shipped=float(Lp.mean()), mean_len_arm=float(Lo.mean()),
                         median_len_shipped=float(np.median(Lp)), median_len_arm=float(np.median(Lo)),
                         ctrl_ecfp_pair_diffs=ctrlT, ctrl_anongraph_pair_diffs=ctrlS,
                         secs=round(time.time() - t0, 1)))
        r = rows[-1]
        print("[%2d/30] %-18s %-10s quar=%4d (%.2f%%) strings %6.2f%% -> labels %6.3f%% "
              "(J=%.3f, +%d/-%d) ctrlT/S=%d/%d"
              % (k + 1, ds, arm, nq, r['quarantined_pct'], r['strings_changed_pct'],
                 r['flip_pct'], r['jaccard'], r['gain'], r['loss'], ctrlT, ctrlS), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "E6_crosstoolkit.csv"), index=False)
    if quar:
        pd.DataFrame(quar).to_csv(os.path.join(C.OUT, "E6_crosstoolkit_quarantine.csv"), index=False)

o = pd.DataFrame(rows)
o.to_csv(os.path.join(C.OUT, "E6_crosstoolkit.csv"), index=False)
if quar:
    pd.DataFrame(quar).to_csv(os.path.join(C.OUT, "E6_crosstoolkit_quarantine.csv"), index=False)

# --------------------------------------------------------------- summary
LOG = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

W("=" * 116)
W("E1  CROSS-SERIALIZATION / CROSS-TOOLKIT ARM")
W("=" * 116)
W("toolkits: RDKit %s | Indigo %s | Open Babel %s"
  % (__import__('rdkit').__version__, INDIGO_VERSION, OB_VERSION))
W("ARM B (ChEMBL canonical_smiles) DECLINED: the distributed files carry no ChEMBL identifier;")
W("      structure-based re-identification would introduce ambiguous entity matching.")
W("")
W("Identity discipline: a rewritten string is used only if RDKit canonical SMILES AND InChIKey both match")
W("the original molecule. Failures are QUARANTINED (shipped string retained) and never counted as label change.")
W("")
hdr = ("%-11s %8s %8s %10s %10s %8s %8s %9s %9s %8s"
       % ("arm", "quar_n", "quar_%", "strings_%", "labels_%", "Jaccard", "cliffs", "gain", "loss", "ctrlT/S"))
W(hdr); W("-" * len(hdr))
base_tot = int(o[o.arm == ARMS[0][0]].n_cliff_base.sum())
summary_rows = []
for arm, _, desc, _ in ARMS:
    a = o[o.arm == arm]
    cw = (a.flip_pct * a.n).sum() / a.n.sum()          # compound(molecule)-weighted
    W("%-11s %8d %8.3f %10.2f %10.3f %8.3f %8d %9d %9d %8s"
      % (arm, int(a.quarantined.sum()), 100 * a.quarantined.sum() / a.n.sum(),
         a.strings_changed_pct.median(), a.flip_pct.median(), a.jaccard.median(),
         int(a.n_cliff_new.sum()), int(a.gain.sum()), int(a.loss.sum()),
         "%d/%d" % (int(a.ctrl_ecfp_pair_diffs.sum()), int(a.ctrl_anongraph_pair_diffs.sum()))))
    W("            %s" % desc)
    W("            label change: median %.3f%%  range %.3f-%.3f%%  molecule-weighted mean %.3f%%"
      % (a.flip_pct.median(), a.flip_pct.min(), a.flip_pct.max(), cw))
    W("            cliff count %d -> %d (%+.2f%%); gains exceed losses in %d/30 datasets"
      % (base_tot, int(a.n_cliff_new.sum()),
         100 * (a.n_cliff_new.sum() - a.n_cliff_base.sum()) / a.n_cliff_base.sum(),
         int((a.gain > a.loss).sum())))
    clean = a[a.quarantined == 0]
    W("            CLEAN SUBSET (zero quarantine): %d/30 datasets, label change median %.3f%% (range %.3f-%.3f%%)"
      % (len(clean), clean.flip_pct.median() if len(clean) else np.nan,
         clean.flip_pct.min() if len(clean) else np.nan,
         clean.flip_pct.max() if len(clean) else np.nan))
    W("            mean SMILES length shipped %.1f -> arm %.1f chars"
      % (a.mean_len_shipped.mean(), a.mean_len_arm.mean()))
    W("")
    summary_rows.append(dict(arm=arm, description=desc,
                             quarantined=int(a.quarantined.sum()),
                             quarantined_pct=100 * a.quarantined.sum() / a.n.sum(),
                             strings_changed_median=a.strings_changed_pct.median(),
                             flip_median=a.flip_pct.median(), flip_min=a.flip_pct.min(),
                             flip_max=a.flip_pct.max(), flip_molweighted=cw,
                             jaccard_median=a.jaccard.median(),
                             n_cliff_base=base_tot, n_cliff_new=int(a.n_cliff_new.sum()),
                             cliff_count_change_pct=100 * (a.n_cliff_new.sum() - a.n_cliff_base.sum()) / a.n_cliff_base.sum(),
                             gain=int(a.gain.sum()), loss=int(a.loss.sum()),
                             n_clean_datasets=len(clean),
                             flip_median_clean=clean.flip_pct.median() if len(clean) else np.nan,
                             ctrl_ecfp_pair_diffs=int(a.ctrl_ecfp_pair_diffs.sum()),
                             ctrl_anongraph_pair_diffs=int(a.ctrl_anongraph_pair_diffs.sum())))
pd.DataFrame(summary_rows).to_csv(os.path.join(C.OUT, "E6_crosstoolkit_summary.csv"), index=False)

W("POSITIVE CONTROL: total ECFP4 pair differences %d, anonymised-graph pair differences %d (must be 0)"
  % (int(o.ctrl_ecfp_pair_diffs.sum()), int(o.ctrl_anongraph_pair_diffs.sum())))
W("")
W("INTERPRETATION KEY (fixed in the study protocol before running):")
W("  cross-toolkit change >= Kekule (5.23%)  -> practical concern STRENGTHENS")
W("  cross-toolkit change <  Kekule          -> narrow the practical claim; Kekule remains the practical result")
W("  cross-toolkit change ~ 0                -> retain the formal non-invariance result and state plainly")
W("                                             that the tested cross-toolkit routes show little empirical divergence")
open(os.path.join(C.OUT, "E6_crosstoolkit.txt"), "w").write("\n".join(LOG))
print("\nwrote E6_crosstoolkit.{csv,txt}, E6_crosstoolkit_summary.csv")
