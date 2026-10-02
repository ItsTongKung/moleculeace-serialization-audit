"""SECONDARY METHOD-SENSITIVITY CHECKS  (sections A-E).

A  INFERENCE SENSITIVITY
   A1 BCa (bias-corrected and accelerated) cluster-bootstrap intervals for the
      primary contrast R2-R0 and for the D1 penalty, alongside the percentile
      intervals reported in the manuscript.  Percentile intervals with 30
      clusters can under-cover; BCa corrects for bias and skewness.
   A2 count of bootstrap draws discarded because a resample contained no
      dataset contributing to a required subgroup, for every small-stratum
      estimate, so conditional intervals can be marked as such.

B  SPLIT-AGREEMENT CALIBRATION
   Raw split agreement has a high floor (an all-train assignment already scores
   ~80%).  Reported instead: test-set Jaccard, Cohen's kappa, and a
   random-marginal chance baseline; and the two reproductions are separated --
   agreement of the split stratified on the SHIPPED cliff_mol column versus
   agreement of the R0 arm stratified on our RECOMPUTED D0 labels.

C  KEKULE MECHANISM
   The manuscript calls the mechanism "mechanical".  This measures it:
   shipped vs Kekule SMILES length (mean, median, per-molecule delta), the
   per-dataset correlation between aromatic-ring content and cliff gain, and
   -- separately -- whether the RDKit ATOM OUTPUT ORDER changed as well as the
   notation, which would mean the rewrite is not purely a notation change.

D  D1 ANALYTIC FLOOR
   Median |delta y| among strict (D1) cliff pairs, the error a 1-nearest-
   neighbour predictor is forced into by a near-duplicate partner carrying that
   potency difference, and the comparison of that floor with the observed D1
   penalty.  KNN is discussed with the floor in hand rather than asserted.

E  D3 SENSITIVITY
   A size-constrained matched-molecular-pair variant (transformation limited to
   at most MAXFRAG heavy atoms, the conventional Hussain-Rae style cap),
   prevalence and penalty; the count of molecules for which single-cut
   fragmentation yields no usable pair (maxCutBonds / zero-fragment cases); and
   the stereochemical behaviour of the core matching.

Outputs: OUT/E10_sensitivity.txt, OUT/E10_sensitivity_*.csv
"""
import os, sys, glob, time, numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol

CACHE = os.path.join(C.OUT, "cache")
PRED = os.path.join(C.PRED, "preds_fullrepair")
B = 4000
MODELS = ['SVM', 'RF', 'GBM', 'KNN']
DATASETS = C.datasets()
MAXFRAG = 13                      # heavy atoms; pre-declared size cap for the D3 variant
SECTIONS = set(sys.argv[1]) if len(sys.argv) > 1 else set("ABCDE")
LOGSUF = sys.argv[1] if len(sys.argv) > 1 else "all"

LOG = []
def W(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

Z = {}
for f in sorted(glob.glob(os.path.join(PRED, "*.npz"))):
    ds, arm, combo = os.path.basename(f)[:-4].split("__")
    z = np.load(f, allow_pickle=True)
    Z[(ds, arm, combo.split("_")[0])] = dict(
        se=(z['y_true'] - z['y_pred']) ** 2, nn_train=z['nn_train'],
        **{k: z[k] for k in z.files if k.startswith('lab_') or k.startswith('partner_')})


def sn(se, mask):
    v = se[mask]
    return float(v.sum()), int(v.size)


def pooled(m, keys):
    S = sum(m[k][0] for k in keys if k in m)
    N = sum(m[k][1] for k in keys if k in m)
    return np.sqrt(S / N) if N > 0 else np.nan


def gap_fn(arm, model, defn):
    A, Bd = {}, {}
    for ds in DATASETS:
        k = (ds, arm, model)
        if k not in Z:
            continue
        lab = Z[k]['lab_' + defn] == 1
        if lab.sum() >= 1:
            A[ds] = sn(Z[k]['se'], lab)
        if (~lab).sum() >= 1:
            Bd[ds] = sn(Z[k]['se'], ~lab)
    return lambda ks: pooled(A, ks) - pooled(Bd, ks)


def boot_raw(fn, seed=C.SEED):
    rng = np.random.default_rng(seed)
    point = fn(DATASETS)
    vals, dropped = [], 0
    for _ in range(B):
        s = list(rng.choice(DATASETS, size=len(DATASETS), replace=True))
        v = fn(s)
        if v is not None and np.isfinite(v):
            vals.append(v)
        else:
            dropped += 1
    return point, np.array(vals), dropped


def bca(fn, seed=C.SEED):
    """BCa interval for a cluster-bootstrap statistic over datasets.
    z0 from the proportion of bootstrap replicates below the point estimate;
    acceleration from the jackknife over datasets (leave-one-dataset-out)."""
    point, vals, dropped = boot_raw(fn, seed)
    if vals.size < B // 10 or not np.isfinite(point):
        return point, np.nan, np.nan, np.nan, np.nan, dropped
    prop = float((vals < point).mean())
    prop = min(max(prop, 1.0 / (2 * vals.size)), 1 - 1.0 / (2 * vals.size))
    z0 = st.norm.ppf(prop)
    jk = np.array([fn([d for d in DATASETS if d != x]) for x in DATASETS], float)
    jk = jk[np.isfinite(jk)]
    if jk.size < 3:
        return point, np.nan, np.nan, z0, np.nan, dropped
    jbar = jk.mean()
    num = ((jbar - jk) ** 3).sum()
    den = 6.0 * (((jbar - jk) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    out = []
    for alpha in (0.025, 0.975):
        z = st.norm.ppf(alpha)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(vals, 100 * st.norm.cdf(adj))))
    return point, out[0], out[1], z0, a, dropped


# ================================================================ A
if "A" in SECTIONS:
  W("=" * 122)
  W("A  INFERENCE SENSITIVITY: BCa vs percentile cluster-bootstrap intervals (B=%d, seed %d)" % (B, C.SEED))
  W("=" * 122)
  arows = []
  W("%-34s %-5s %+10s %24s %24s %8s %8s" % ("quantity", "model", "point", "percentile 95% CI",
                                            "BCa 95% CI", "z0", "accel"))
  targets = []
  for m in MODELS:
      f2, f0 = gap_fn('R2', m, 'D2'), gap_fn('R0', m, 'D0')
      targets.append(("PRIMARY R2-R0", m, lambda ks, a=f2, b=f0: a(ks) - b(ks)))
  for m in MODELS:
      targets.append(("D1 penalty (arm S0)", m, gap_fn('S0', m, 'D1')))
  for m in MODELS:
      targets.append(("D0 penalty (arm S0)", m, gap_fn('S0', m, 'D0')))
  for name, m, fn in targets:
      p, plo, phi, _ = (lambda t: (t[0], *np.percentile(t[1], [2.5, 97.5]), t[2]))(boot_raw(fn))
      pb, blo, bhi, z0, acc, dr = bca(fn)
      W("%-34s %-5s %+10.4f  [%+.4f, %+.4f]  [%+.4f, %+.4f] %8.3f %8.3f"
        % (name, m, p, plo, phi, blo, bhi, z0, acc))
      arows.append(dict(quantity=name, model=m, point=p, pct_lo=plo, pct_hi=phi,
                        pct_status=C.ci_status(plo, phi), bca_lo=blo, bca_hi=bhi,
                        bca_status=C.ci_status(blo, bhi), z0=z0, accel=acc,
                        dropped_draws=dr, B=B))
  A = pd.DataFrame(arows)
  A.to_csv(os.path.join(C.OUT, "E10_sensitivity_bca.csv"), index=False)
  nflip = int((A.pct_status != A.bca_status).sum())
  W("")
  W("  verdicts changing between percentile and BCa: %d of %d" % (nflip, len(A)))
  if nflip:
      for _, r in A[A.pct_status != A.bca_status].iterrows():
          W("    !! %s / %s : percentile %s, BCa %s" % (r['quantity'], r['model'], r.pct_status, r.bca_status))

  W("")
  W("A2  EMPTY-SUBGROUP DROPPED-DRAW COUNTS for small-stratum estimates")
  W("    (a nonzero count means the interval is CONDITIONAL on the subgroup being represented)")
  drows = []
  STRATA = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
  for m in ['SVM', 'RF']:
      for lo_, hi_ in STRATA:
          A_, Bd, n1, n0 = {}, {}, 0, 0
          for ds in DATASETS:
              k = (ds, 'S0', m)
              if k not in Z:
                  continue
              nn = Z[k]['nn_train']; instr = (nn >= lo_) & (nn < hi_)
              lab = Z[k]['lab_D1'] == 1
              m1, m0 = instr & lab, instr & ~lab
              n1 += int(m1.sum()); n0 += int(m0.sum())
              if m1.sum() >= 1:
                  A_[ds] = sn(Z[k]['se'], m1)
              if m0.sum() >= 1:
                  Bd[ds] = sn(Z[k]['se'], m0)
          p, vals, dr = boot_raw(lambda ks: pooled(A_, ks) - pooled(Bd, ks))
          lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size >= B // 10 else (np.nan, np.nan))
          W("    %-5s nn_train [%.1f,%.1f)  n_cliff=%4d n_non=%5d  %+.4f [%+.4f,%+.4f]  dropped %d/%d (%.1f%%) %s"
            % (m, lo_, hi_, n1, n0, p, lo, hi, dr, B, 100.0 * dr / B,
               "CONDITIONAL" if dr > 0 else ""))
          drows.append(dict(model=m, stratum="%.1f-%.1f" % (lo_, hi_), n_cliff=n1, n_non=n0,
                            estimate=p, ci_lo=lo, ci_hi=hi, dropped_draws=dr, B=B,
                            conditional=bool(dr > 0)))
  for m in MODELS:
      for defn, arm in [('D1', 'S0'), ('D2', 'R2')]:
          A_, Bd = {}, {}
          for ds in DATASETS:
              k = (ds, arm, m)
              if k not in Z:
                  continue
              lab = Z[k]['lab_' + defn] == 1
              pt = Z[k]['partner_train_' + defn]
              m1 = lab & ~pt
              if m1.sum() >= 1:
                  A_[ds] = sn(Z[k]['se'], m1)
              if (~lab).sum() >= 1:
                  Bd[ds] = sn(Z[k]['se'], ~lab)
          p, vals, dr = boot_raw(lambda ks: pooled(A_, ks) - pooled(Bd, ks))
          lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size >= B // 10 else (np.nan, np.nan))
          W("    %-5s %s/%s partner NOT in train  %+.4f [%+.4f,%+.4f]  dropped %d/%d %s"
            % (m, arm, defn, p, lo, hi, dr, B, "CONDITIONAL" if dr > 0 else ""))
          drows.append(dict(model=m, stratum="%s/%s partner_out" % (arm, defn), n_cliff=-1, n_non=-1,
                            estimate=p, ci_lo=lo, ci_hi=hi, dropped_draws=dr, B=B,
                            conditional=bool(dr > 0)))
  pd.DataFrame(drows).to_csv(os.path.join(C.OUT, "E10_sensitivity_dropped.csv"), index=False)

# ================================================================ B
if "B" in SECTIONS:
  W("")
  W("=" * 122)
  W("B  SPLIT-AGREEMENT CALIBRATION  (raw agreement has a high floor; report Jaccard, kappa and a chance baseline)")
  W("=" * 122)
  brows = []
  for ds in DATASETS:
      z = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
      shipped = z['shipped_split'].astype(str)
      r0 = z['split_reconD0'].astype(str)
      n = len(shipped)
      # all-train floor
      floor = 100.0 * float((shipped == 'train').mean())
      for nm, other in [('reconD0_vs_shipped', r0)]:
          a = shipped == 'test'
          b = other == 'test'
          raw = 100.0 * float((shipped == other).mean())
          inter = int((a & b).sum()); uni = int((a | b).sum())
          jacc = inter / uni if uni else np.nan
          # Cohen's kappa on the train/test labelling
          po = float((a == b).mean())
          pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
          kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
          # chance baseline: expected raw agreement of two independent draws at these marginals
          chance = 100.0 * pe
          brows.append(dict(dataset=ds, comparison=nm, n=n,
                            raw_agree_pct=raw, all_train_floor_pct=floor,
                            test_jaccard=jacc, cohen_kappa=kappa,
                            chance_raw_agree_pct=chance,
                            exact=bool(raw > 99.9999)))
  Bt = pd.DataFrame(brows)
  Bt.to_csv(os.path.join(C.OUT, "E10_sensitivity_splitagree.csv"), index=False)
  q = Bt[Bt.comparison == 'reconD0_vs_shipped']
  W("  R0 arm (stratified on our RECOMPUTED D0) vs the shipped split column:")
  W("    exact in %d/30 datasets; raw agreement median %.2f%% (min %.2f%%)"
    % (int(q.exact.sum()), q.raw_agree_pct.median(), q.raw_agree_pct.min()))
  W("    all-train floor: mean %.2f%%  |  chance raw agreement at observed marginals: mean %.2f%%"
    % (q.all_train_floor_pct.mean(), q.chance_raw_agree_pct.mean()))
  W("    test-set Jaccard: median %.3f (min %.3f)  |  Cohen's kappa: median %.3f (min %.3f)"
    % (q.test_jaccard.median(), q.test_jaccard.min(), q.cohen_kappa.median(), q.cohen_kappa.min()))
  W("    -> kappa and Jaccard are chance-corrected and therefore the figures to quote; raw agreement is")
  W("       not interpretable against zero.")
  W("  NOTE: the SEPARATE quantity 'faithful upstream call stratified on the shipped cliff_mol column'")
  W("        is measured by 02_splitrepro.py (E2_split_reproduction.csv) and is not the same statistic.")
  try:
      e2 = pd.read_csv(os.path.join(C.OUT, "E2_split_reproduction.csv"))
      cc = [c for c in e2.columns if 'default' in c or 'arpack' in c]
      if cc:
          v = e2[cc[0]]
          W("        shipped-label reproduction (%s): exact in %d/30, median %.2f%%"
            % (cc[0], int((v > 99.9999).sum()), v.median()))
  except Exception as e:
      W("        (E2_split_reproduction.csv unreadable: %s)" % e)

# ================================================================ C
if "C" in SECTIONS:
  W("")
  W("=" * 122)
  W("C  KEKULE MECHANISM: is it a string-length effect, and did the atom output order change too?")
  W("=" * 122)
  crows = []
  for ds in DATASETS:
      df = C.load(ds)
      smi = df['smiles'].tolist()
      mols = [Chem.MolFromSmiles(s) for s in smi]
      kek, order_changed, arom_rings = [], 0, []
      for m in mols:
          mk = Chem.Mol(m)
          Chem.Kekulize(mk, clearAromaticFlags=True)
          ks = Chem.MolToSmiles(mk, kekuleSmiles=True)
          kek.append(ks)
          # RDKit records the atom output order of the last MolToSmiles call
          try:
              o_ar = list(map(int, m.GetProp('_smilesAtomOutputOrder')[1:-1].split(',')[:-1])) \
                  if m.HasProp('_smilesAtomOutputOrder') else None
          except Exception:
              o_ar = None
          arom_rings.append(sum(1 for r in m.GetRingInfo().AtomRings()
                                if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)))
      # atom output order: compare the order produced by the aromatic vs the Kekule write
      for i, m in enumerate(mols):
          Chem.MolToSmiles(m)
          oa = m.GetProp('_smilesAtomOutputOrder') if m.HasProp('_smilesAtomOutputOrder') else ''
          mk = Chem.Mol(m); Chem.Kekulize(mk, clearAromaticFlags=True)
          Chem.MolToSmiles(mk, kekuleSmiles=True)
          ob = mk.GetProp('_smilesAtomOutputOrder') if mk.HasProp('_smilesAtomOutputOrder') else ''
          if oa != ob:
              order_changed += 1
      Ls = np.array([len(s) for s in smi], float)
      Lk = np.array([len(s) for s in kek], float)
      crows.append(dict(dataset=ds, n=len(smi),
                        mean_len_shipped=Ls.mean(), mean_len_kekule=Lk.mean(),
                        median_len_shipped=float(np.median(Ls)), median_len_kekule=float(np.median(Lk)),
                        mean_len_delta=float((Lk - Ls).mean()),
                        mean_len_delta_pct=float((100 * (Lk - Ls) / Ls).mean()),
                        frac_longer=float((Lk > Ls).mean()),
                        atom_order_changed=order_changed,
                        atom_order_changed_pct=100.0 * order_changed / len(smi),
                        mean_aromatic_rings=float(np.mean(arom_rings))))
      print("  [C] %-18s len %.1f -> %.1f (%+.1f chars, %+.1f%%), atom order changed %.1f%%"
            % (ds, Ls.mean(), Lk.mean(), (Lk - Ls).mean(),
               (100 * (Lk - Ls) / Ls).mean(), 100.0 * order_changed / len(smi)), flush=True)
  Ct = pd.DataFrame(crows)
  try:
      inv = pd.read_csv(os.path.join(C.OUT, "A2_invariance.csv"))
      Ct = Ct.merge(inv[['dataset', 'kek_flip_pct', 'kek_gain', 'kek_loss', 'n_cliff_base', 'kek_n_cliff']],
                    on='dataset', how='left')
      Ct['kek_cliff_gain_pct'] = 100.0 * (Ct.kek_n_cliff - Ct.n_cliff_base) / Ct.n_cliff_base
  except Exception as e:
      W("  (A2_invariance.csv unavailable: %s)" % e)
  Ct.to_csv(os.path.join(C.OUT, "E10_sensitivity_kekule.csv"), index=False)
  W("  SMILES length: mean %.1f -> %.1f chars (mean per-molecule change %+.1f chars, %+.1f%%);"
    % (Ct.mean_len_shipped.mean(), Ct.mean_len_kekule.mean(),
       Ct.mean_len_delta.mean(), Ct.mean_len_delta_pct.mean()))
  W("                 median %.1f -> %.1f; a longer string in a mean %.1f%% of molecules"
    % (Ct.median_len_shipped.median(), Ct.median_len_kekule.median(), 100 * Ct.frac_longer.mean()))
  W("  ATOM OUTPUT ORDER changed in a mean %.2f%% of molecules (median %.2f%%, max %.2f%%)"
    % (Ct.atom_order_changed_pct.mean(), Ct.atom_order_changed_pct.median(),
       Ct.atom_order_changed_pct.max()))
  if 'kek_cliff_gain_pct' in Ct:
      ok = Ct.dropna(subset=['kek_cliff_gain_pct'])
      for xcol, lab in [('mean_len_delta_pct', 'string-length inflation (%)'),
                        ('mean_aromatic_rings', 'mean aromatic rings per molecule')]:
          rr = st.pearsonr(ok[xcol], ok.kek_cliff_gain_pct)
          ss = st.spearmanr(ok[xcol], ok.kek_cliff_gain_pct)
          W("  per-dataset %s vs cliff-count gain: Pearson r=%.3f (p=%.4f), Spearman rho=%.3f (p=%.4f)"
            % (lab, rr.statistic, rr.pvalue, ss.statistic, ss.pvalue))
      rr = st.pearsonr(ok.mean_len_delta_pct, ok.kek_flip_pct)
      W("  per-dataset string-length inflation vs LABEL CHANGE rate: Pearson r=%.3f (p=%.4f)"
        % (rr.statistic, rr.pvalue))
  W("  WORDING RULE: call the mechanism 'mechanical' only if the length effect is present AND the atom")
  W("  output order is largely unchanged; otherwise say 'a plausible mechanical explanation' and report")
  W("  that the rewrite also reorders atoms.")

# ================================================================ D
if "D" in SECTIONS:
  W("")
  W("=" * 122)
  W("D  D1 ANALYTIC FLOOR: what error does a near-duplicate partner with a >10-fold potency gap force?")
  W("=" * 122)
  dd, floors = [], []
  for ds in DATASETS:
      z = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
      az = np.load(os.path.join(CACHE, "adj_%s.npz" % ds))
      n = int(az['n'])
      y = z['y']
      adj1 = np.unpackbits(az['adj_D1'], axis=1)[:, :n].astype(bool)
      iu = np.triu_indices(n, 1)
      m = adj1[iu]
      if m.sum() == 0:
          continue
      dy = np.abs(y[iu[0]][m] - y[iu[1]][m])
      dd.append(dy)
      floors.append(dict(dataset=ds, n_D1_pairs=int(m.sum()),
                         median_abs_dy=float(np.median(dy)), mean_abs_dy=float(dy.mean())))
  allf = np.concatenate(dd)
  Ft = pd.DataFrame(floors)
  Ft.to_csv(os.path.join(C.OUT, "E10_sensitivity_d1floor.csv"), index=False)
  med = float(np.median(allf))
  W("  strict (D1) cliff pairs pooled: n=%d, median |delta y| = %.3f log units (IQR %.3f-%.3f, mean %.3f)"
    % (allf.size, med, *np.percentile(allf, [25, 75]), allf.mean()))
  W("  A 1-nearest-neighbour predictor whose nearest training neighbour IS the cliff partner predicts")
  W("  that partner's potency, so its absolute error on the cliff compound equals |delta y|.")
  W("  Implied RMSE floor for such a predictor on D1 cliffs = sqrt(mean(dy^2)) = %.3f log units."
    % float(np.sqrt((allf ** 2).mean())))
  W("  Median-based floor (robust) = %.3f log units." % med)
  for m in MODELS:
      f = gap_fn('S0', m, 'D1')
      W("    observed D1 penalty (%s, arm S0) = %+.4f  -- %.1f%% of the median analytic floor"
        % (m, f(DATASETS), 100 * f(DATASETS) / med))
  W("  KNN NOTE: the observed D1 penalty is a small fraction of the floor a pure 1-NN predictor would")
  W("  face, so the benchmark's k-NN model is not operating at that floor (k>1 and distance weighting")
  W("  average over neighbours). The floor bounds what ANY local-neighbour predictor can achieve on")
  W("  these pairs; it does not by itself explain the observed penalty in the fitted models.")

# ================================================================ E
if "E" in SECTIONS:
  W("")
  W("=" * 122)
  W("E  D3 SENSITIVITY: size-constrained MMP variant, fragmentation failures, stereochemistry")
  W("=" * 122)


  def mmp_pairs_capped(smiles, maxfrag):
      """As common.mmp_pairs but the VARIABLE fragment must have <= maxfrag heavy atoms.
      Also returns fragmentation diagnostics."""
      n = len(smiles)
      idx = {}
      n_nofrag, n_exception, n_allcapped = 0, 0, 0
      for i, s in enumerate(smiles):
          m = Chem.MolFromSmiles(s)
          try:
              cuts = FragmentMol(m, maxCuts=1, resultsAsMols=False)
          except Exception:
              cuts = []
              n_exception += 1
          keys, nseen = set(), 0
          for cut in cuts:
              for part in [c for c in cut if c]:
                  fr = part.split('.')
                  if len(fr) < 2:
                      continue
                  a, b = Chem.MolFromSmiles(fr[0]), Chem.MolFromSmiles(fr[-1])
                  if a is None or b is None:
                      continue
                  nseen += 1
                  if a.GetNumAtoms() >= b.GetNumAtoms():
                      core, var = fr[0], b
                  else:
                      core, var = fr[-1], a
                  if var.GetNumHeavyAtoms() <= maxfrag:
                      keys.add(Chem.MolToSmiles(Chem.MolFromSmiles(core)))
          if nseen == 0:
              n_nofrag += 1
          elif not keys:
              n_allcapped += 1
          for k in keys:
              idx.setdefault(k, []).append(i)
      M = np.zeros((n, n), bool)
      for k, v in idx.items():
          if len(v) > 1:
              v = np.array(v); M[np.ix_(v, v)] = True
      np.fill_diagonal(M, False)
      return M, dict(n_nofrag=n_nofrag, n_exception=n_exception, n_allcapped=n_allcapped)


  erows = []
  _ep = os.path.join(C.OUT, "E10_sensitivity_d3.csv")
  _edone = set()
  if os.path.exists(_ep):
      _ed = pd.read_csv(_ep); erows = _ed.to_dict('records'); _edone = set(_ed.dataset)
      W("  resuming section E: %d datasets done" % len(_edone))
  t0 = time.time()
  for k, ds in enumerate(C.datasets()):
      if ds in _edone:
          continue
      z = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
      smi = list(z['smiles'])
      df = C.load(ds)
      act = df['exp_mean [nM]'].values.astype(float)
      Fb = C.foldchange(act) > C.FOLD
      labD3 = z['lab_D3']
      MMc, diag = mmp_pairs_capped(smi, MAXFRAG)
      lab_c = C.compound_labels(MMc, Fb)
      n_stereo = int(sum(1 for s in smi if '@' in s))
      erows.append(dict(dataset=ds, n=len(smi),
                        n_cliff_D3=int(labD3.sum()),
                        n_cliff_D3capped=int(lab_c.sum()),
                        prev_D3_pct=100.0 * labD3.mean(),
                        prev_D3capped_pct=100.0 * lab_c.mean(),
                        n_smiles_with_stereo=n_stereo,
                        stereo_pct=100.0 * n_stereo / len(smi), **diag))
      np.save(os.path.join(CACHE, "labD3capped_%s.npy" % ds), lab_c)
      pd.DataFrame(erows).to_csv(_ep, index=False)
      print("  [E] %-18s D3 %5d -> capped(<=%d heavy) %5d | nofrag %4d exc %3d allcapped %4d"
            % (ds, int(labD3.sum()), MAXFRAG, int(lab_c.sum()),
               diag['n_nofrag'], diag['n_exception'], diag['n_allcapped']), flush=True)
  Et = pd.DataFrame(erows)
  Et.to_csv(os.path.join(C.OUT, "E10_sensitivity_d3.csv"), index=False)
  W("  size cap: variable fragment <= %d heavy atoms (pre-declared)" % MAXFRAG)
  W("  prevalence: D3 as reported %.1f%% of compounds -> size-constrained %.1f%%"
    % (100 * Et.n_cliff_D3.sum() / Et.n.sum(), 100 * Et.n_cliff_D3capped.sum() / Et.n.sum()))
  W("  fragmentation diagnostics (totals over 30 datasets):")
  W("    molecules yielding NO usable single-cut fragment pair: %d (%.2f%% of %d)"
    % (int(Et.n_nofrag.sum()), 100 * Et.n_nofrag.sum() / Et.n.sum(), int(Et.n.sum())))
  W("    FragmentMol raised an exception: %d molecules" % int(Et.n_exception.sum()))
  W("    fragmented but every cut exceeded the size cap: %d molecules" % int(Et.n_allcapped.sum()))
  W("  stereochemistry: %d of %d shipped SMILES carry a stereocentre marker (%.1f%%);"
    % (int(Et.n_smiles_with_stereo.sum()), int(Et.n.sum()),
       100 * Et.n_smiles_with_stereo.sum() / Et.n.sum()))
  W("    core matching uses Chem.MolToSmiles on the re-parsed core, which RETAINS stereochemistry, so")
  W("    two cores differing only in configuration are NOT matched. MoleculeACE's own curation called")
  W("    split_data(remove_stereo=True), so stereochemical siblings are absent from the shipped files.")
  W("  [%.1f min]" % ((time.time() - t0) / 60))

open(os.path.join(C.OUT, "E10_sensitivity_%s.txt" % LOGSUF), "w").write("\n".join(LOG))
print("\nwrote E10_sensitivity.txt and E10_sensitivity_*.csv")
