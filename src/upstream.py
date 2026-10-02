"""Loads the genuine upstream MoleculeACE modules WITHOUT importing the package
__init__ (which pulls in torch / tensorflow).  Gives access to the real
cliffs.py and data_prep.py source as distributed."""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(os.environ.get("MOLECULEACE_DIR", os.path.join(_ROOT, "external", "MoleculeACE")),
                    "MoleculeACE")


def _stub(name):
    m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m; return m


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "MoleculeACE.benchmark.cliffs" not in sys.modules:
    _stub("MoleculeACE"); _stub("MoleculeACE.benchmark")
    const = _load("MoleculeACE.benchmark.const", os.path.join(_PKG, "benchmark", "const.py"))
    cliffs = _load("MoleculeACE.benchmark.cliffs", os.path.join(_PKG, "benchmark", "cliffs.py"))
    data_prep = _load("MoleculeACE.benchmark.data_prep", os.path.join(_PKG, "benchmark", "data_prep.py"))
else:
    const = sys.modules["MoleculeACE.benchmark.const"]
    cliffs = sys.modules["MoleculeACE.benchmark.cliffs"]
    data_prep = sys.modules["MoleculeACE.benchmark.data_prep"]

RANDOM_SEED = const.RANDOM_SEED
ActivityCliffs = cliffs.ActivityCliffs
get_tanimoto_matrix = cliffs.get_tanimoto_matrix
get_scaffold_matrix = cliffs.get_scaffold_matrix
get_levenshtein_matrix = cliffs.get_levenshtein_matrix
moleculeace_similarity = cliffs.moleculeace_similarity
split_data = data_prep.split_data
