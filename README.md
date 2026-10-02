# moleculeace-serialization-audit

Code, derived data and per-molecule predictions for the preprint

> Sriwicha P. **SMILES serialization changes activity-cliff annotations in the MoleculeACE
> benchmark.** ChemRxiv (2026). Preprint DOI: *assigned by ChemRxiv on posting*.

Archived release of this repository: [doi:10.5281/zenodo.23096971](https://doi.org/10.5281/zenodo.23096971) (Zenodo, version 1.0.0).

## 1. What is tested

[MoleculeACE](https://github.com/molML/MoleculeACE) labels a compound as an activity-cliff compound when
it forms a pair with a >10-fold potency difference and at least one of three similarities reaches
0.9: ECFP4 Tanimoto, Tanimoto on anonymized molecular graphs, or normalized Levenshtein similarity
between SMILES strings. The first two are computed on the parsed molecule; the third is computed on
the SMILES strings as supplied, without canonicalization. This study holds molecules, potencies,
thresholds and code fixed and varies only how the same molecules are written (serialized), then
follows the consequences into the train/test split and into retrained models.

## 2. Main result

| Rewrite of the same molecules | Median share of compound labels that change (30 datasets) |
|---|---|
| RDKit canonical round-trip (negative control) | 0.00% |
| RDKit Kekulé canonical SMILES | 5.23% |
| **Indigo canonical SMILES** (independent toolkit) | **8.98%** |
| **Open Babel canonical SMILES** (independent toolkit) | **9.48%** |
| Randomized SMILES (adversarial) | 17.05% |

Every rewritten string was checked to encode the original molecule (RDKit canonical SMILES and
InChIKey). The two graph-derived criteria were invariant under every rewrite. Because MoleculeACE
stratifies its split on the cliff label, changing the labels reassigned 31.3% (median) of compounds
between train and test, close to the 32.21% expected for an independent redraw. Repairing the
benchmark end to end (serialization-invariant labels, re-derived split, retrained SVM/RF/GBM/KNN
models) moved the pooled cliff penalty by −0.006 to +0.020 log units, with every 95% interval
including zero (minimum detectable effect 0.041–0.057). Details and limitations are in the preprint.

## 3. Data used

Only the public MoleculeACE benchmark: 30 ChEMBL-derived datasets (48,714 rows, 35,633 unique
SMILES), its hyperparameter configuration files and its distributed results table. These files are
**not redistributed here**; `reproduce.py fetch` clones them from the audited commit.

## 4. Audited upstream commit

`molML/MoleculeACE` at **`7e6de0bd2968c56589c580f2a397f01c531ede26`** (2025-02-15; the corrected
benchmark release). MIT license.

## 5. Quick start

Tested on Linux with CPython 3.12 in a fresh virtual environment.

```bash
git clone https://github.com/ItsTongKung/moleculeace-serialization-audit.git
cd moleculeace-serialization-audit
python3.12 -m venv .venv && source .venv/bin/activate     # or: conda env create -f environment.yml
pip install -r requirements.txt
python reproduce.py all-quick
```

`all-quick` runs three stages (about 20 minutes on two CPU cores):

| Stage | What it does | Expected output |
|---|---|---|
| `fetch` | clones MoleculeACE into `external/` and checks out the audited commit | `upstream MoleculeACE at 7e6de0b…: 30 benchmark datasets` |
| `figures` | redraws Figures 1–4 from `results/summary/` | `figure source data: 10/10 files identical to the released copies` |
| `verify` | runs the fidelity gate against the upstream code, recomputes labels and splits from the upstream files, recomputes the primary statistics from the released predictions | `split assignments: 30/30 …`, `split realizations: 30/30 …`, `label sets: 30/30 …`, `primary statistics …: identical`, `verify: OK` |

`python reproduce.py full` re-runs the entire pipeline including all model fits (about 5 hours on two
cores) and writes everything to `outputs/`. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for
the run order, run times and known reproduction limits (in particular: KNN predictions are not
reproducible across machines; see §4 there).

## 6. Software

Python 3.12; RDKit 2026.03.6, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.0,
RapidFuzz 3.14.6, matplotlib 3.11.1; EPAM Indigo 1.46.0 and Open Babel 3.1.0 (`openbabel-wheel`
3.1.1.23) for the cross-toolkit arm only. Exact pins: [requirements.txt](requirements.txt).

## 7. Repository layout

```
reproduce.py          entry point (fetch / figures / verify / full)
src/                  shared primitives (common.py) and the loader for the upstream code (upstream.py)
scripts/              analysis scripts, numbered in execution order (01-21)
labels/               30 files: per-compound cliff labels under D0 (shipped definition), D1, D2, D3
splits/               30 files: shipped, reconstructed (R0) and repaired (R2) splits, clusters,
                      nearest-training-neighbor similarity
splits/realizations/  30 files: the 24 same-definition split realizations (matched control)
predictions/          per-molecule test predictions of every fitted model (2,520 files; see docs)
results/summary/      every analysis result table (CSV) and text summary
figures/              Figures 1-4 (PDF, PNG); figures/source_data/ holds the data behind every panel
docs/                 METHODS.md, REPRODUCIBILITY.md, OUTPUTS.md (file-by-file guide)
```

Definitions: D0, as shipped (ECFP4 ∨ anonymized graph ∨ string); D1, ECFP4 only; D2, ECFP4 ∨
anonymized graph (the minimal serialization-invariant repair); D3, single-cut matched molecular pair.

## 8. Bulk prediction artifacts

All per-molecule predictions are in `predictions/` (about 25 MB) and in the archived release:
`preds_fullrepair/` (360 files: 30 datasets × 3 split arms × 4 models, ECFP4),
`preds_alt/` (720 files: MACCS and physicochemical descriptors), and
`preds_splitnoise/` (1,440 files: 24 split realizations × 30 datasets, SVM, plus the split vectors).
Each `.npz` holds `y_true`, `y_pred`, `smiles` and the cliff labels; see [docs/OUTPUTS.md](docs/OUTPUTS.md).

## 9. Preprint

ChemRxiv: link and DOI will be added here once the preprint is posted.

## 10. Licenses

* **Code** (`src/`, `scripts/`, `reproduce.py`): MIT, see [LICENSE-CODE](LICENSE-CODE).
* **Derived data** (label annotations, split assignments, result tables, figure source data,
  predictions, figures): CC BY 4.0, see [LICENSE-DATA](LICENSE-DATA).
* **Upstream content.** The `smiles`, potency (`y_pEC50_pKi`, `exp_mean_nM`) and shipped-label
  (`cliff_mol_shipped`, `split_shipped`) columns reproduce values distributed by MoleculeACE
  (MIT license, © 2022 Derek van Tilborg), whose bioactivity data originate from ChEMBL
  (CC BY-SA 3.0). Those columns remain under their original terms; the CC BY 4.0 grant covers only
  the annotations and results produced by this study. MoleculeACE itself is not redistributed.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the preprint and, for the data, the archived release.
