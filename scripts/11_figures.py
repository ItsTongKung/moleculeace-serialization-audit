"""Regenerates Figures 1-4 from the CURRENT analysis outputs.

No number is typed by hand: every plotted value is read from a saved analysis
table in the output directory (common.OUT), and the exact frame that backs each
panel is written out as FIGDATA_*.csv (common.FIGDATA).

Figure 1 panel (c) shows the branch operating regions on two yardsticks that are
not MoleculeACE branches (MACCS-key Tanimoto and an MCS-based similarity).
Figure 3 panel (c) includes the magnitude-matched split-realization control
(17_splitnoise.py, SVM only) and the DESIGN contrast (full repair minus
fixed-split relabeling, 16_paired.py).  Legacy column names such as scaf_only
and pct_reach_scaf denote the anonymized-graph branch (see docs/OUTPUTS.md).
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common as C

FIG = C.FIGDIR
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7,
                     'ytick.major.width': 0.7, 'savefig.dpi': 400, 'figure.dpi': 140,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})
CE, CS, CL = '#1b4b8f', '#c8721a', '#7a2f8f'      # ecfp / anonymized-graph / string
CA = ['#4c5f6b', '#2c7fb8', '#b03a2e', '#2e7d32', '#8e44ad']
C_IND, C_OB, C_KEK, C_RT = '#0f7b6c', '#8a6d1f', '#b03a2e', '#2e7d32'
MODS = ['SVM', 'RF', 'GBM', 'KNN']
wid = 0.2


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print("wrote", name)


def rd(fn):
    p = os.path.join(C.OUT, fn)
    return pd.read_csv(p) if os.path.exists(p) else None


# =================================================================== FIG 1
comp = rd("A6_composition.csv")
hist = rd("A4_branch_histogram.csv")
yh = rd("E7_yardstick_hist.csv")
yo = rd("E7_yardstick_ordering.csv")
ncol = 3 if yh is not None else 2
fig, ax = plt.subplots(1, ncol, figsize=(3.7 * ncol, 3.0))

o = comp.sort_values('pct_reach_str').reset_index(drop=True)
x = np.arange(len(o))
ax[0].plot(x, o.pct_reach_str, 'o', ms=3.2, color=CL, label='SMILES-Levenshtein $\\geq$ 0.9')
ax[0].plot(x, o.pct_reach_scaf, 's', ms=3.2, color=CS, label='anonymized graph $\\geq$ 0.9')
ax[0].plot(x, o.pct_reach_ecfp, '^', ms=3.2, color=CE, label='ECFP4 Tanimoto $\\geq$ 0.9')
for c, col in [('pct_reach_str', CL), ('pct_reach_scaf', CS), ('pct_reach_ecfp', CE)]:
    ax[0].axhline(o[c].median(), color=col, lw=0.8, ls='--', alpha=0.55)
ax[0].set_xlabel('dataset (sorted)')
ax[0].set_ylabel('% of cliff compounds reachable\nby that branch alone')
ax[0].set_ylim(-3, 118)
ax[0].legend(frameon=False, fontsize=6.4, loc='upper left', ncol=1, handletextpad=0.3)
ax[0].set_title('a  Which branch reaches each cliff compound', loc='left', fontsize=8.5,
                fontweight='bold')

mid = (hist.bin_lo + hist.bin_hi) / 2
w = (hist.bin_hi - hist.bin_lo).iloc[0]
for col, c, lab in [('str_only', CL, 'string-only (n=%d)' % hist.str_only.sum()),
                    ('scaf_only', CS, 'anonymized-graph-only (n=%d)' % hist.scaf_only.sum()),
                    ('ecfp_admitted', CE, 'ECFP4 branch (n=%d)' % hist.ecfp_admitted.sum())]:
    ax[1].bar(mid, 100 * hist[col] / hist[col].sum(), width=w, color=c, alpha=0.55,
              label=lab, edgecolor='none')
ax[1].axvline(0.9, color='k', lw=1.0, ls=':')
ax[1].text(0.885, ax[1].get_ylim()[1] * 0.93, 'ECFP4-branch\nthreshold = 0.9', ha='right',
           va='top', fontsize=6.4)
ax[1].set_xlabel('ECFP4 Tanimoto similarity of the cliff pair')
ax[1].set_ylabel("% of that branch's pairs")
ax[1].legend(frameon=False, fontsize=6.4, loc='upper left')
ax[1].set_title('b  Operating region on ECFP4 (branch (a) itself)', loc='left', fontsize=8.5,
                fontweight='bold')

if yh is not None:
    ymid = (yh.bin_lo + yh.bin_hi) / 2
    yw = (yh.bin_hi - yh.bin_lo).iloc[0]
    off = {'maccs': -0.0, 'mcs': 0.0}
    styles = {'maccs': dict(alpha=0.55, hatch=None), 'mcs': dict(alpha=0.0, hatch='///')}
    for ruler, ls in [('maccs', '-'), ('mcs', '--')]:
        for cat, c in [('str_only', CL), ('scaf_only', CS), ('ecfp_admitted', CE)]:
            col = "%s_%s" % (ruler, cat)
            if col not in yh:
                continue
            v = 100 * yh[col] / max(yh[col].sum(), 1)
            ax[2].plot(ymid, v, ls, color=c, lw=1.2 if ruler == 'maccs' else 1.0,
                       alpha=0.95 if ruler == 'maccs' else 0.75)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=CL, lw=1.2, label='string-only'),
           Line2D([], [], color=CS, lw=1.2, label='anonymized-graph-only'),
           Line2D([], [], color=CE, lw=1.2, label='ECFP4 branch'),
           Line2D([], [], color='k', lw=1.2, ls='-', label='MACCS Tanimoto'),
           Line2D([], [], color='k', lw=1.0, ls='--', label='MCS similarity')]
    ax[2].legend(handles=leg, frameon=False, fontsize=6.0, loc='upper left')
    ax[2].set_xlabel('similarity on a NON-branch yardstick')
    ax[2].set_ylabel("% of that branch's pairs")
    txt = []
    for _, r in yo.iterrows():
        txt.append("%s: $\\Delta$=%+.3f [%+.3f, %+.3f]" % (r.ruler.upper(), r.delta_median,
                                                           r.ci_lo, r.ci_hi))
    ax[2].text(0.02, 0.34, "median(anon-graph-only) $-$ median(string-only)\n" + "\n".join(txt),
               transform=ax[2].transAxes, fontsize=5.8, va='top')
    ax[2].set_title('c  Same comparison, non-branch yardsticks', loc='left', fontsize=8.5,
                    fontweight='bold')
save(fig, "Fig1_definition")
comp.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig1a.csv"), index=False)
hist.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig1b.csv"), index=False)
if yh is not None:
    yh.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig1c.csv"), index=False)

# =================================================================== FIG 2
inv = rd("A2_invariance.csv").set_index("dataset")
pm = rd("A2b_invariance_permol.csv").set_index("dataset")
# The randomized arm plotted here is the PRIMARY per-molecule seeding scheme; the shared-seed
# values are retained in the frame under a shared_ prefix so both are traceable in one file.
for c in ["rand_flip_mean_pct", "rand_flip_sd_pct", "rand_jaccard_mean", "rand_n_cliff_mean",
          "rand_flip_min_pct", "rand_flip_max_pct", "ever_flip_pct"]:
    inv["shared_" + c] = inv[c]
    inv[c] = pm[c]
inv = inv.reset_index()

ct = rd("E6_crosstoolkit.csv")
if ct is not None:
    for arm, pref in [('indigo', 'ind'), ('openbabel', 'ob')]:
        a = ct[ct.arm == arm].set_index('dataset')
        inv = inv.merge(a[['flip_pct', 'jaccard', 'n_cliff_new', 'quarantined_pct']]
                        .rename(columns={'flip_pct': pref + '_flip_pct',
                                         'jaccard': pref + '_jaccard',
                                         'n_cliff_new': pref + '_n_cliff',
                                         'quarantined_pct': pref + '_quarantined_pct'}),
                        left_on='dataset', right_index=True, how='left')

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
o = inv.sort_values('rand_flip_mean_pct').reset_index(drop=True)
x = np.arange(len(o))
ax[0].bar(x, o.rand_flip_mean_pct, yerr=o.rand_flip_sd_pct, color='#9fb6cd', edgecolor='none',
          error_kw=dict(lw=0.6, capsize=1.4), label='randomized SMILES (10 seeds)')
if 'ind_flip_pct' in o:
    ax[0].plot(x, o.ind_flip_pct, 's', ms=3.0, color=C_IND, label='Indigo canonical')
    ax[0].plot(x, o.ob_flip_pct, '^', ms=3.0, color=C_OB, label='Open Babel canonical')
ax[0].plot(x, o.kek_flip_pct, 'D', ms=3.0, color=C_KEK, label='Kekulé canonical')
ax[0].plot(x, o.rt_flip_pct, 'o', ms=2.4, color=C_RT, label='canonical round-trip')
ax[0].axhline(0, color='k', lw=0.8)
ax[0].set_xlabel('dataset (sorted)')
ax[0].set_ylabel('% of compound labels that change')
ax[0].legend(frameon=False, fontsize=6.0, loc='upper left')
ax[0].set_title('a  Label change under rewriting', loc='left', fontsize=8.2, fontweight='bold')

ax[1].plot(x, o.rand_jaccard_mean, 'o', ms=3.2, color='#5a6f8c', label='randomized')
if 'ind_jaccard' in o:
    ax[1].plot(x, o.ind_jaccard, 's', ms=3.0, color=C_IND, label='Indigo')
    ax[1].plot(x, o.ob_jaccard, '^', ms=3.0, color=C_OB, label='Open Babel')
ax[1].plot(x, o.kek_jaccard, 'D', ms=3.0, color=C_KEK, label='Kekulé')
ax[1].plot(x, o.rt_jaccard, 's', ms=2.4, color=C_RT, label='round-trip')
ax[1].set_ylim(0, 1.05)
ax[1].axhline(1.0, color='k', lw=0.7, ls=':')
ax[1].set_xlabel('dataset (same order)')
ax[1].set_ylabel('Jaccard overlap with the shipped cliff set')
ax[1].legend(frameon=False, fontsize=6.2, loc='lower right', ncol=2)
ax[1].set_title('b  Set overlap', loc='left', fontsize=8.2, fontweight='bold')

labels = ['aromatic\n(as shipped)', 'Kekulé']
tot = [inv.n_cliff_base.sum(), inv.kek_n_cliff.sum()]
cols = ['#4c5f6b', C_KEK]
if 'ind_n_cliff' in inv:
    labels += ['Indigo', 'Open\nBabel']
    tot += [inv.ind_n_cliff.sum(), inv.ob_n_cliff.sum()]
    cols += [C_IND, C_OB]
labels += ['randomized\n(mean of 10)']
tot += [inv.rand_n_cliff_mean.sum()]
cols += ['#9fb6cd']
bars = ax[2].bar(labels, tot, color=cols, width=0.62)
for b, v in zip(bars, tot):
    ax[2].text(b.get_x() + b.get_width() / 2, v + max(tot) * 0.012, "%d" % round(v),
               ha='center', fontsize=6.4)
ax[2].axhline(inv.n_cliff_base.sum(), color='k', lw=0.7, ls=':')
ax[2].set_ylabel('total cliff-labeled compounds\n(same %d molecules)' % int(inv.n.sum()))
ax[2].set_ylim(0, max(tot) * 1.16)
ax[2].tick_params(axis='x', labelsize=6.4)
ax[2].set_title('c  Cliff-set size by serialization', loc='left', fontsize=8.2, fontweight='bold')
save(fig, "Fig2_invariance")
inv.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig2.csv"), index=False)

# =================================================================== FIG 3
res = rd("A1_analysis_results.csv")
lad = res[res.analysis == 'ladder']
arm = res[res.analysis == 'arm_penalty']
con = res[res.analysis == 'contrast']
pair = rd("E8_paired.csv")
sn = rd("E9_splitnoise.csv")

fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.2))
DEFO = ['D3', 'D0', 'D2', 'D1']
for i, m in enumerate(MODS):
    s = lad[lad.model == m].set_index('definition').reindex(DEFO)
    pos = np.arange(len(DEFO)) + (i - 1.5) * wid
    ax[0].bar(pos, s.estimate, wid, yerr=[s.estimate - s.ci_lo, s.ci_hi - s.estimate],
              error_kw=dict(lw=0.6, capsize=1.4), label=m, edgecolor='none', alpha=0.9)
ax[0].axhline(0, color='k', lw=0.8)
ax[0].set_xticks(np.arange(len(DEFO)))
ax[0].set_xticklabels(['D3\nMMP', 'D0\nshipped', 'D2\ninvariant', 'D1\nstrict'], fontsize=6.8)
ax[0].set_ylabel('cliff penalty (log units)\nRMSE(cliff) $-$ RMSE(own non-cliff)')
ax[0].legend(frameon=False, fontsize=6.4, ncol=2)
ax[0].set_title('a  Definitional ladder, shipped benchmark', loc='left', fontsize=8.2,
                fontweight='bold')

for i, m in enumerate(MODS):
    s = arm[arm.model == m].set_index('arm').reindex(['S0', 'R0', 'R2'])
    pos = np.arange(3) + (i - 1.5) * wid
    ax[1].bar(pos, s.estimate, wid, yerr=[s.estimate - s.ci_lo, s.ci_hi - s.estimate],
              error_kw=dict(lw=0.6, capsize=1.4), label=m, edgecolor='none', alpha=0.9)
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_xticks(np.arange(3))
ax[1].set_xticklabels(['S0\nshipped split\n(D0)', 'R0\nreconstructed\n(D0)',
                       'R2\nfull repair\n(D2, retrained)'], fontsize=6.6)
ax[1].set_ylabel('cliff penalty (log units)')
ax[1].set_title('b  Full-repair counterfactual', loc='left', fontsize=8.2, fontweight='bold')

# panel c: every contrast bootstrapped directly, clearly separated by kind
rows = []
for nm, tag, col in [('PRIMARY  full repair      R2 - R0', 'PRIMARY', CA[2]),
                     ('CONTROL  split drift      R0 - S0', 'CONTROL', CA[1]),
                     ('BRIDGE   shipped->repair  R2 - S0', 'BRIDGE', CA[0])]:
    for m in MODS:
        r = con[(con.contrast == nm) & (con.model == m)]
        if len(r):
            r = r.iloc[0]
            rows.append((tag, m, r.estimate, r.ci_lo, r.ci_hi, col))
if pair is not None:
    for m in MODS:
        r = pair[(pair.analysis == 'A_design_difference') & (pair.model == m)]
        if len(r):
            r = r.iloc[0]
            rows.append(('DESIGN', m, r.estimate, r.ci_lo, r.ci_hi, CA[4]))
if sn is not None:
    for _, r in sn.iterrows():
        # the matched control is a spread, not a signed contrast: plot +/- p95 of the pairwise
        # |difference| between same-definition split realizations, centered on zero
        rows.append(('MATCHED', r.model, 0.0, -r.pairwise_abs_p95, r.pairwise_abs_p95, CA[3]))
ypos, labs = [], []
k = 0
last = None
for tag, m, est, lo, hi, col in rows:
    if last is not None and tag != last:
        k += 0.8
    ax[2].errorbar(est, k, xerr=[[est - lo], [hi - est]], fmt='o', ms=3.0, lw=0.9,
                   color=col, capsize=1.6)
    labs.append("%-7s %s" % (tag, m)); ypos.append(k); k += 1; last = tag
ax[2].axvline(0, color='k', lw=0.8, ls='--')
ax[2].set_yticks(ypos)
ax[2].set_yticklabels(labs, fontsize=5.2, family='monospace')
ax[2].set_xlabel('difference in cliff penalty (log units)')
ax[2].invert_yaxis()
ax[2].set_title('c  Paired contrasts, 95% CI', loc='left', fontsize=8.2, fontweight='bold')
save(fig, "Fig3_definition")
pd.DataFrame(rows, columns=['contrast', 'model', 'estimate', 'ci_lo', 'ci_hi', '_color']) \
    .drop(columns='_color').to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig3c.csv"), index=False)
lad.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig3a.csv"), index=False)
arm.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig3b.csv"), index=False)

# =================================================================== FIG 4
dec = rd("A5_deconfound_pooled.csv")
par = res[res.analysis == 'partner_location']
prox = res[res.analysis == 'proximity']
fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.2))
DESC = ['ECFP', 'MACCS', 'PHYSCHEM']
if dec is not None:
    for i, m in enumerate(MODS):
        s = dec[(dec.model == m) & (dec.definition == 'D1') & (dec.arm == 'S0')] \
            .set_index('descriptor').reindex(DESC)
        pos = np.arange(len(DESC)) + (i - 1.5) * wid
        ax[0].bar(pos, s.estimate, wid, yerr=[s.estimate - s.ci_lo, s.ci_hi - s.estimate],
                  error_kw=dict(lw=0.6, capsize=1.4), label=m, edgecolor='none', alpha=0.9)
    ax[0].set_xticks(np.arange(len(DESC)))
    ax[0].set_xticklabels(['ECFP4', 'MACCS', 'physico-\nchemical'], fontsize=6.8)
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].legend(frameon=False, fontsize=6.4, ncol=2, loc='upper right', bbox_to_anchor=(1.0, 0.87))
    if pair is not None:
        pm_ = pair[(pair.analysis == 'C_ecfp_minus_maccs') & (pair.note == 'S0/D1')]
        pp_ = pair[(pair.analysis == 'D_ecfp_minus_physchem') & (pair.note == 'S0/D1')]
        if len(pm_) and len(pp_):
            ax[0].set_ylim(0, 0.78)
            ax[0].text(0.02, 0.98,
                       "paired ECFP4 $-$ MACCS: %d/%d CI exclude 0\n"
                       "paired ECFP4 $-$ physchem: %d/%d CI exclude 0"
                       % (int((pm_.ci_status == 'EXCLUDES 0').sum()), len(pm_),
                          int((pp_.ci_status == 'EXCLUDES 0').sum()), len(pp_)),
                       transform=ax[0].transAxes, fontsize=5.8, va='top')
ax[0].set_ylabel('strict-cliff (D1) penalty (log units)')
ax[0].set_title('a  Shared-representation control', loc='left', fontsize=8.2, fontweight='bold')

sub = par[(par.definition == 'D1') & (par.arm == 'S0')]
for i, m in enumerate(MODS):
    r = sub[sub.model == m]
    if not len(r):
        continue
    r = r.iloc[0]
    for j, (v, lo, hi, c) in enumerate([(r.gap_in, r.in_lo, r.in_hi, '#b03a2e'),
                                        (r.gap_out, r.out_lo, r.out_hi, '#4c5f6b'),
                                        (r.estimate, r.ci_lo, r.ci_hi, '#2c7fb8')]):
        ax[1].errorbar(j + (i - 1.5) * 0.16, v, yerr=[[v - lo], [hi - v]], fmt='o', ms=3.0,
                       lw=0.9, color=c, capsize=1.6)
ax[1].axhline(0, color='k', lw=0.8, ls='--')
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['partner in\ntraining set', 'partner not in\ntraining set',
                       'difference\n(directly estimated)'], fontsize=6.4)
ax[1].set_ylabel('penalty vs non-cliffs (log units)')
ax[1].set_title('b  Partner location, D1 (4 model families)', loc='left', fontsize=8.2,
                fontweight='bold')

drop = rd("E10_sensitivity_dropped.csv")
for i, m in enumerate(['SVM', 'RF']):
    s = prox[prox.model == m].reset_index(drop=True)
    pos = np.arange(len(s)) + (i - 0.5) * 0.3
    ax[2].bar(pos, s.estimate, 0.3, yerr=[s.estimate - s.ci_lo, s.ci_hi - s.estimate],
              error_kw=dict(lw=0.6, capsize=1.4), label=m, edgecolor='none', alpha=0.9)
    ax[2].set_xticks(np.arange(len(s)))
    ax[2].set_xticklabels(["%s\n(n=%d)" % (a, b) for a, b in zip(s.stratum, s.n_cliff)],
                          fontsize=6.4)
ax[2].axhline(0, color='k', lw=0.8)
if drop is not None:
    cond = drop[(drop.conditional) & (drop.stratum.astype(str).str.startswith('0.7'))]
    if len(cond):
        pct = 100.0 * cond.iloc[0].dropped_draws / cond.iloc[0].B
        ax[2].text(0.02, 0.97,
                   "lowest stratum: conditional interval\n(%.1f%% of bootstrap draws dropped)" % pct,
                   transform=ax[2].transAxes, fontsize=5.8, va='top')
ax[2].set_xlabel('nearest-neighbor ECFP4 similarity to the training set')
ax[2].set_ylabel('D1 penalty within stratum')
ax[2].legend(frameon=False, fontsize=6.6)
ax[2].set_title('c  Within-proximity-stratum comparison', loc='left', fontsize=8.2,
                fontweight='bold')
save(fig, "Fig4_controls")
if dec is not None:
    dec.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig4a.csv"), index=False)
par.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig4b.csv"), index=False)
prox.to_csv(os.path.join(C.FIGDATA, "FIGDATA_Fig4c.csv"), index=False)
print("figures done")
