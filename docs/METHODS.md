# Methods summary

This is a compact guide to what each analysis does and where its code and outputs are. The full
description is in the Methods section of the preprint and in Supplementary Information, Section S1.

## Object of study

MoleculeACE (`molML/MoleculeACE`, commit `7e6de0bd2968c56589c580f2a397f01c531ede26`): 30 ChEMBL
datasets, 48,714 rows, 35,633 unique SMILES. Upstream modules `benchmark/cliffs.py`,
`benchmark/data_prep.py` and `benchmark/const.py` are loaded directly (`src/upstream.py`) for the
fidelity gate.

## Cliff definitions

A pair is a cliff pair if the potency fold change exceeds 10 and the similarity criterion holds; a
compound is cliff-labeled if it belongs to at least one cliff pair.

| Label | Similarity criterion | Serialization-invariant |
|---|---|---|
| D0 | ECFP4 Tanimoto ≥ 0.9 OR anonymized-graph Tanimoto ≥ 0.9 OR normalized Levenshtein ≥ 0.9 (as shipped) | no |
| D1 | ECFP4 Tanimoto ≥ 0.9 | yes |
| D2 | ECFP4 OR anonymized graph (D0 without the string criterion) | yes |
| D3 | single-cut matched molecular pair with a shared canonical core | yes |

"Anonymized graph" is RDKit `MakeScaffoldGeneric` applied to the whole molecule (all heavy atoms to
carbon, all bonds single, side chains retained). It is not a Bemis–Murcko framework; the upstream
function is named `get_scaffold_matrix`, and legacy column names in this repository use `scaf`.

## Arms of the end-to-end repair

| Arm | Split stratified on | Evaluated under |
|---|---|---|
| S0 | shipped `split` column | D0 (recomputed) |
| R0 | split re-derived from recomputed D0 labels | D0 |
| R2 | split re-derived from D2 labels; models retrained | D2 |

Primary contrast: R2 − R0. Drift control: R0 − S0. The cliff penalty is RMSE on cliff compounds
minus RMSE on the arm's own non-cliff complement, pooled over datasets; intervals are 95% percentile
intervals from a cluster bootstrap over the 30 datasets (B = 4,000, seed 42).

## Analysis status

| Analysis | Script | Status |
|---|---|---|
| Fidelity gate against upstream code | `01_fidelity.py` | specified before execution |
| Label and split reproduction | `02_splitrepro.py`, `03_cache.py` | exploratory |
| Criterion decomposition, composition, size bias | `07_branches.py`, `10_composition.py` | exploratory |
| Serialization invariance (round-trip, Kekulé, randomized) | `06_invariance.py`, `06b_invariance_permol.py` | exploratory; seeded rerun added after the initial analysis |
| Fixed-split model re-evaluation; split perturbation | `08_analysis.py` | specified in the study protocol before execution; outcome changed from cliff-RMSE minus overall RMSE to the own-complement gap (protocol deviation) |
| Own-complement baseline, three controls, ladder decomposition | `08_analysis.py`, `09_deconfound.py`, `12_deconfound_analysis.py` | post hoc |
| Full-repair counterfactual (S0, R0, R2) | `05_fullrepair.py`, `08_analysis.py` | post hoc |
| Branch operating regions | `07_branches.py` | post hoc |
| Cross-toolkit serialization (Indigo, Open Babel, InChI) | `14_crosstoolkit.py` | specified in writing before execution, after initial results |
| Non-branch yardsticks (MACCS, MCS) | `15_yardstick.py` | specified in writing before execution, after initial results |
| Direct paired contrasts | `16_paired.py` | specified in writing before execution, after initial results |
| Magnitude-matched split control (SVM only) | `17_splitnoise.py` | specified in writing before execution, after initial results |
| Sensitivity checks (BCa, dropped draws, split agreement, Kekulé mechanism, D1 floor, MMP size cap) | `18_sensitivity.py`, `20_d3capped.py` | specified in writing before execution, after initial results |
| KNN distance-tie diagnosis | `19_knnties.py` | post hoc diagnosis |

None of the analyses was registered on a public registry. A mechanistic prediction (that the string
criterion favors larger molecules) was stated in the study protocol before the analysis was run and
was not supported by the data.

## Cross-toolkit identity checks

A rewritten string is used only if it parses back (RDKit) to a molecule whose canonical SMILES and
InChIKey both match the original. Molecules that do not pass are quarantined: they keep their shipped
string and are counted as toolkit round-trip failures, never as label changes
(`results/summary/E6_crosstoolkit_quarantine.csv`). A ChEMBL-identifier route was not used because
the distributed benchmark files carry no molecule identifiers, and re-deriving them by structure
search would introduce ambiguous entity matching.
