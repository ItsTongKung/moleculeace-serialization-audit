#!/usr/bin/env python3
"""Reproduction entry point for the MoleculeACE serialization audit.

Stages (run one at a time, or `all-quick`):

  fetch     Clone MoleculeACE at the audited commit into external/MoleculeACE.
  figures   Redraw Figures 1-4 from the released result tables and check that the
            regenerated figure source data are identical to figures/source_data/.
            (~1 min)
  verify    Recompute the analysis cache from the upstream benchmark files, re-export
            the released label sets and split assignments, recompute the pooled
            primary statistics from the released per-molecule predictions, and compare
            every regenerated file with the released copy.  (~15-20 min on 2 CPU cores)
  all-quick fetch + figures + verify.
  full      Re-run the complete pipeline from scratch, including model fitting
            (about 5 h on 2 CPU cores).  Outputs go to outputs/; compare with results/.

Every stage writes only below outputs/ (or the directory given by MSA_OUT) and never
modifies the released files.
"""
import os, sys, subprocess, filecmp, glob, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_URL = "https://github.com/molML/MoleculeACE.git"
UPSTREAM_COMMIT = "7e6de0bd2968c56589c580f2a397f01c531ede26"
UPSTREAM = os.environ.get("MOLECULEACE_DIR", os.path.join(ROOT, "external", "MoleculeACE"))
OUT = os.environ.get("MSA_OUT", os.path.join(ROOT, "outputs"))
PY = sys.executable

FULL_ORDER = ["01_fidelity", "02_splitrepro", "03_cache", "04_adjacency", "05_fullrepair",
              "06_invariance", "06b_invariance_permol", "07_branches", "09_deconfound",
              "10_composition", "08_analysis", "12_deconfound_analysis", "13_published_fidelity",
              "14_crosstoolkit", "15_yardstick", "19_knnties", "16_paired",
              ("18_sensitivity", "ABCDE"), "20_d3capped", "17_splitnoise", "11_figures",
              "21_export_resources"]


def sh(args, env=None):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(args, env=env)
    if r.returncode:
        sys.exit("FAILED: " + " ".join(args))


def run(script, *argv, **envextra):
    env = dict(os.environ, MOLECULEACE_DIR=UPSTREAM, MSA_OUT=OUT)
    env.update(envextra)
    sh([PY, os.path.join(ROOT, "scripts", script + ".py"), *argv], env=env)


def fetch():
    if not os.path.isdir(os.path.join(UPSTREAM, ".git")):
        os.makedirs(os.path.dirname(UPSTREAM), exist_ok=True)
        sh(["git", "clone", "--quiet", UPSTREAM_URL, UPSTREAM])
    sh(["git", "-C", UPSTREAM, "checkout", "--quiet", UPSTREAM_COMMIT])
    head = subprocess.check_output(["git", "-C", UPSTREAM, "rev-parse", "HEAD"], text=True).strip()
    assert head == UPSTREAM_COMMIT, head
    n = len(glob.glob(os.path.join(UPSTREAM, "MoleculeACE", "Data", "benchmark_data", "CHEMBL*.csv")))
    assert n == 30, f"expected 30 benchmark files, found {n}"
    print(f"upstream MoleculeACE at {head}: {n} benchmark datasets")


def compare(new_dir, ref_dir, pattern="*.csv", label=""):
    new = sorted(glob.glob(os.path.join(new_dir, pattern)))
    if not new:
        sys.exit(f"FAILED: no regenerated files in {new_dir}")
    bad = [os.path.basename(f) for f in new
           if not filecmp.cmp(f, os.path.join(ref_dir, os.path.basename(f)), shallow=False)]
    print(f"{label}: {len(new) - len(bad)}/{len(new)} files identical to the released copies")
    return bad


def figures():
    tmp = os.path.join(OUT, "figure_check")
    os.makedirs(tmp, exist_ok=True)
    run("11_figures", MSA_OUT=os.path.join(ROOT, "results", "summary"),
        MSA_FIGDIR=os.path.join(tmp, "figures"), MSA_FIGDATA=os.path.join(tmp, "source_data"))
    bad = compare(os.path.join(tmp, "source_data"), os.path.join(ROOT, "figures", "source_data"),
                  label="figure source data")
    if bad:
        sys.exit(f"FAILED: figure source data differ: {bad}")
    print("figures: OK (rendered to %s)" % os.path.join(tmp, "figures"))


def verify():
    pred = os.path.join(ROOT, "predictions")
    run("01_fidelity")
    run("02_splitrepro")
    run("03_cache")
    run("21_export_resources", MSA_PRED=pred)
    res = os.path.join(OUT, "resources")
    bad = []
    bad += compare(os.path.join(res, "splits"), os.path.join(ROOT, "splits"), label="split assignments")
    bad += compare(os.path.join(res, "splits", "realizations"), os.path.join(ROOT, "splits", "realizations"),
                   label="split realizations")
    # label files: the size-capped D3 column needs 18_sensitivity.py section E (slow);
    # compare every other column exactly
    import pandas as pd
    lab_bad = []
    for f in sorted(glob.glob(os.path.join(res, "labels", "*.csv"))):
        a = pd.read_csv(f)
        b = pd.read_csv(os.path.join(ROOT, "labels", os.path.basename(f)))
        cols = [c for c in a.columns if c in b.columns]
        if not a[cols].equals(b[cols]):
            lab_bad.append(os.path.basename(f))
    print(f"label sets: {30 - len(lab_bad)}/30 identical on all recomputed columns")
    run("08_analysis", MSA_PRED=pred)
    a = pd.read_csv(os.path.join(OUT, "A1_analysis_results.csv"))
    b = pd.read_csv(os.path.join(ROOT, "results", "summary", "A1_analysis_results.csv"))
    same = a.shape == b.shape and a.round(10).equals(b.round(10))
    print("primary statistics (A1_analysis_results.csv):", "identical" if same else "DIFFERENT")
    if bad or lab_bad or not same:
        sys.exit(f"FAILED: {bad + lab_bad}{'' if same else ' + A1_analysis_results.csv'}")
    print("verify: OK")


def full():
    for item in FULL_ORDER:
        if isinstance(item, tuple):
            run(item[0], *item[1:])
        else:
            run(item)
    print("full pipeline finished; outputs in", OUT)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all-quick"
    os.makedirs(OUT, exist_ok=True)
    if stage == "fetch":
        fetch()
    elif stage == "figures":
        figures()
    elif stage == "verify":
        verify()
    elif stage == "all-quick":
        fetch(); figures(); verify()
    elif stage == "full":
        fetch(); full()
    else:
        sys.exit(__doc__)
