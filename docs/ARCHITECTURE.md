# Architecture

| Location | Role |
| --- | --- |
| `src/gastric_mtl/cox.py` | Cox objective, smoothed Group Lasso, directional coupling and concordance |
| `src/gastric_mtl/solver.py` | Stable directional objective and strict Newton refinement |
| `src/gastric_mtl/second_order.py` | Hessian products and task/group conditioning |
| `src/gastric_mtl/warmup.py` | Conditioned L-BFGS warmup |
| `src/gastric_mtl/quadratic.py`, `proximal.py` | Alternative sharing utilities for method comparisons |
| `src/gastric_mtl/data.py`, `eac.py` | Cohort alignment, XML parsing, histology restriction and training-only preprocessing |
| `src/gastric_mtl/metrics.py` | Survival probabilities and training-derived IPCW evaluation |
| `src/gastric_mtl/paths.py` | Central workspace configuration and source correspondence checks |
| `workflows/experiments/` | Fixed study protocols, molecular fitting and clinical augmentation |
| `workflows/validation/` | Objective, prediction, partition and selection checks |
| `workflows/analysis/` | Prediction aggregation and conditional uncertainty |
| `workflows/figures/` | Summary-driven result figures |
| `scripts/` | Workflow dispatch, dependency reporting, integrity checks and source packaging |

Workflow stages pass structured cohort objects, candidate specifications and prediction records. Data preparation aligns patients and endpoints; training folds determine preprocessing parameters and candidate selection; validators reconstruct fitted objectives and predictions; analysis computes aggregate metrics and paired conditional intervals.

Numerical function definitions are checked against syntax-tree fingerprints. Source correspondence records distinguish exact byte identity from a reviewed relocation. Training freezes current source and input hashes in its protocol and checks them before fitting.

The command runner supports spawn-based multiprocessing. Data locations are independent of the shell's working directory. Protocol checks precede preflight output, and incompatible run directories are rejected to prevent mixing experiments.
