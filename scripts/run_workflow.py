#!/usr/bin/env python3
"""One command-line entry point for the maintained research workflows."""

import argparse
import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    "prepare-data": ("gastric_mtl.data", "Build the aligned TCGA cohort cache from local source data."),
    "initial-evaluation": ("workflows.experiments.initial_evaluation", "Generate the original single-task references and patient splits."),
    "train-three": ("workflows.experiments.three_cancer", "Fit the strict ESCA/STAD/COAD single, pair and triple comparisons."),
    "preflight-eac": ("workflows.data.preflight_eac", "Check EAC/STAD splits and freeze the protocol before training."),
    "train-eac": ("workflows.experiments.eac_stad", "Fit the EAC/STAD single and pair comparison."),
    "audit-eac": ("workflows.validation.audit_eac_cohorts", "Reconcile EAC/STAD cohorts with source clinical XML."),
    "validate-three": ("workflows.validation.three_cancer", "Recompute three-cancer predictions, objectives and selection."),
    "validate-eac": ("workflows.validation.eac_stad", "Recompute EAC/STAD predictions, objectives and selection."),
    "summarize-three": ("workflows.analysis.three_cancer", "Aggregate validated three-cancer folds and historical comparisons."),
    "summarize-eac": ("workflows.analysis.eac_stad", "Aggregate validated EAC/STAD folds and uncertainty intervals."),
    "figures-three": ("workflows.figures.three_cancer", "Render three-cancer result figures from validated summaries."),
    "figures-eac": ("workflows.figures.eac_stad", "Render EAC/STAD framework and result figures."),
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", type=Path, help="Parent directory containing code/, datasets/ and results/.")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List available workflows without importing scientific libraries.")
    for name, (_, description) in WORKFLOWS.items():
        command = sub.add_parser(name, help=description, description=description)
        if name in ("train-three", "train-eac"):
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--seed", type=int, choices=[42, 2026])
            command.add_argument("--fold", type=int, choices=range(5))
            command.add_argument("--init-only", action="store_true")
        elif name == "prepare-data":
            command.add_argument("--rebuild", action="store_true")
        elif name == "initial-evaluation":
            command.add_argument("--outer", type=int, default=5)
            command.add_argument("--inner", type=int, default=3)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--max-iter", type=int, default=2000)
            command.add_argument("--name", default="nested_primary")
            command.add_argument("--background", action="store_true")
        elif name.startswith("validate-"):
            command.add_argument("--completed-only", action="store_true")
        elif name == "figures-eac":
            command.add_argument("--framework-only", action="store_true")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.command == "list":
        for name, (_, description) in WORKFLOWS.items():
            print("{:<20} {}".format(name, description))
        return
    if getattr(args, "workers", 1) < 1:
        cli.error("--workers must be positive")
    if sys.flags.optimize:
        cli.error("Do not run with -O: scientific validation uses assertions.")
    if args.workspace:
        os.environ["GASTRIC_MTL_WORKSPACE"] = str(args.workspace.expanduser().resolve())
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gastric_mtl_mpl"))
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    module = WORKFLOWS[args.command][0]
    forwarded = []
    for name, value in vars(args).items():
        if name in ("workspace", "command") or value is None or value is False:
            continue
        forwarded.append("--" + name.replace("_", "-"))
        if value is not True:
            forwarded.append(str(value))
    execute(module, forwarded)


def execute(module, arguments):
    """Expose the real workflow as __main__ for multiprocessing spawn workers."""
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
