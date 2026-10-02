# Output files

File-name prefixes (A1, E6, ...) are stable output identifiers; the table maps each to its script
and to where the result appears in the preprint (SI = Supplementary Information).

| Files in `results/summary/` | Script | Content | Preprint |
|---|---|---|---|
| `E1_fidelity.txt` | `01_fidelity.py` | fidelity gate against upstream `cliffs.py`; metric equivalence | Results 2.1; SI S2 |
| `E2_split_reproduction.csv` | `02_splitrepro.py` | split-reproduction solver scan per dataset | SI S3 |
| `E3_cache_summary.csv` | `03_cache.py` | per-dataset label counts, label and split reproduction | Results 2.1, 2.4 |
| `A1_fullrepair_results.csv` | `05_fullrepair.py` | per-dataset RMSE and cliff RMSE, 3 arms × 4 models | Table 1 |
| `A1_analysis_results.csv`, `A1_analysis.txt` | `08_analysis.py` | pooled penalties, paired contrasts, ladder, controls, Wilcoxon tests, ranks | Tables 1–2; Figure 3 |
| `A2_invariance*.csv`, `A2_invariance_summary.txt` | `06_invariance.py` | round-trip, Kekulé, shared-seed randomized SMILES | Results 2.3; Figure 2 |
| `A2b_invariance_permol*.csv/.txt` | `06b_invariance_permol.py` | per-molecule-seeded randomized SMILES (primary) | Results 2.3; Figure 2 |
| `A4_branch_*` | `07_branches.py` | branch operating regions on ECFP4 | Figure 1b; SI S6 |
| `A5_deconfound_results.csv`, `A5_deconfound_pooled.*` | `09_deconfound.py`, `12_deconfound_analysis.py` | MACCS and physicochemical arms; partner location; proximity strata; null predictor | Figure 4; SI S7, S9 |
| `A6_composition.*`, `A6_size_gradient.csv` | `10_composition.py` | compound-level reachability; size analysis | Figure 1a; SI S11 |
| `E5_published_fidelity.*` | `13_published_fidelity.py` | agreement with distributed MoleculeACE results | Results 2.1 |
| `E6_crosstoolkit*.csv/.txt` | `14_crosstoolkit.py` | Indigo, Open Babel, InChI arms; quarantined molecules | Results 2.3; Figure 2; SI S5 |
| `E7_yardstick*` | `15_yardstick.py` | MACCS and MCS rulers; paired ordering test | Figure 1c; SI S6 |
| `E8_paired.*` | `16_paired.py` | direct paired contrasts A–D | Table 2; SI S7 |
| `E9_splitnoise*` | `17_splitnoise.py` | magnitude-matched split control (SVM) | Results 2.5; Figure 3c; SI S8 |
| `E10_sensitivity_*` | `18_sensitivity.py`, `20_d3capped.py` | BCa, dropped draws, split agreement, Kekulé mechanism, D1 floor, MMP size cap | SI S9, S12 |
| `E11_knnties*` | `19_knnties.py` | KNN distance ties across backends and dtypes | Results 2.1; SI S13 |

`figures/source_data/FIGDATA_Fig<n><panel>.csv` holds the exact data behind each figure panel.

## Per-compound files

* `labels/<dataset>_invariant_labels.csv`: `smiles`, `y_pEC50_pKi`, `exp_mean_nM`,
  `cliff_mol_shipped` (upstream), `D0_recomputed`, `D1_ecfp4_strict`, `D2_invariant`, `D3_mmp`,
  `D3_mmp_size_capped_13` (1 = cliff compound).
* `splits/<dataset>_splits.csv`: `split_shipped` (upstream), `split_R0_from_D0`, `split_R2_from_D2`,
  `cluster_D0`, `cluster_D2`, `nn_ecfp4_sim_to_train_{shipped,R0,R2}` (nearest-training-neighbor
  ECFP4 Tanimoto similarity).
* `splits/realizations/<dataset>_split_realizations.csv`: 24 columns `repNN_stateS`, the train/test
  assignment of each same-definition split realization and its `random_state`.

## Predictions

`predictions/preds_fullrepair/<dataset>__<arm>__<model>_ECFP.npz` (360 files) contain `y_true`,
`y_pred`, `smiles`, `lab_D0`–`lab_D3`, partner-in-train/partner-in-test flags per definition and the
nearest-training-neighbor similarity. `predictions/preds_alt/` (720 files; MACCS and
physicochemical) contain `y_true`, `y_pred`, `smiles`, `lab_D0`–`lab_D3` and identifying metadata.
`predictions/preds_splitnoise/` contains, per dataset and realization, the SVM predictions
(`__repNN__SVM_ECFP.npz`) and the split vector (`__repNN__split.npy`).

## Legacy names

| Name in code or column headers | Meaning |
|---|---|
| `scaf`, `scaf_only`, `pct_reach_scaf`, `get_scaffold_matrix` | the anonymized-graph branch (b) |
| `generic=True` in `common.morgan()` | ECFP4 computed on the anonymized graph |
| `preds_alt` | MACCS and physicochemical descriptor arms |
| `rand_*` columns in `A2_invariance.csv` | shared-seed randomization; the per-molecule scheme is in `A2b_*` |
| `splitnoise` | the magnitude-matched split-realization control |
