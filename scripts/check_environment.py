#!/usr/bin/env python3
"""Report Python and numerical-library versions for the active environment."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

PACKAGES = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
            "scikit-learn": "sklearn", "matplotlib": "matplotlib"}


def inventory():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gastric_mtl_mpl"))
    versions, distributions, errors = {}, {}, {}
    for name, module in PACKAGES.items():
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
        try:
            versions[name] = importlib.import_module(module).__version__
        except Exception as exc:
            versions[name] = None
            errors[name] = str(exc)
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "distribution_metadata": distributions,
        "version_mismatches": {name: {"runtime": versions[name], "metadata": distributions[name]}
                               for name in PACKAGES if versions[name] != distributions[name]},
        "import_errors": errors,
    }


def main():
    result = inventory()
    print(json.dumps(result, indent=2))
    return int(bool(result["import_errors"]))


if __name__ == "__main__":
    sys.exit(main())
