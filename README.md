# Selective sharing in multitask pathway Cox models

Research code for studying when, and how much, survival models should share information across gastrointestinal cancers. The NumPy/SciPy implementation combines Cox partial likelihood, a smoothed Group Lasso pathway penalty and directional task coupling.

## Study scope

| Abbreviation | Cohort |
| --- | --- |
| ESCA | TCGA esophageal carcinoma, including adenocarcinoma and squamous cell carcinoma |
| STAD | TCGA stomach adenocarcinoma |
| COAD | TCGA colon adenocarcinoma |
| EAC | Esophageal adenocarcinoma subset of ESCA, selected by clinical histology |

The implementation supports independent models, all three pairwise combinations and a three-task model for ESCA/STAD/COAD, together with an EAC–STAD comparison. Sharing strength and task combinations are selected using training partitions. Zero sharing is an eligible outcome.

## Repository structure

```text
.
├── src/gastric_mtl/     # Objectives, solvers, cohort processing and survival metrics
├── workflows/          # Experiments, validation, analysis and figures
├── scripts/            # Command-line runner, software checks and source export
├── tests/              # Synthetic numerical and repository tests
├── docs/               # Installation, configuration and methodological notes
├── provenance/         # Source digests and numerical-definition fingerprints
├── pyproject.toml      # Package metadata and dependencies
└── environment.yml     # Reference numerical environment
```

This repository contains source code and technical documentation. Research inputs and generated outputs are managed in an external workspace.

## Installation

Use Python 3.9 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

On Windows, activate with `.venv\Scripts\activate`. The installation is editable; run workflows from the source checkout. `environment.yml` records the reference numerical package versions.

## Quick check

```bash
python -m unittest discover -s tests -v
python scripts/verify_code.py
python scripts/run_workflow.py list
```

The synthetic tests exercise derivatives, optimization, preprocessing isolation, command dispatch and source integrity independently of the research cohorts.

## Running experiments

Configure the workspace and supply inputs according to [Data preparation](docs/DATA.md) and [Reproducibility](docs/REPRODUCIBILITY.md). Inspect the training options with:

```bash
python scripts/run_workflow.py train-three --help
python scripts/run_workflow.py train-eac --help
```

The study uses two seeds (42 and 2026), five outer folds and three inner folds. Workspace configuration is centralized in `src/gastric_mtl/paths.py` and can be set through `--workspace` or `GASTRIC_MTL_WORKSPACE`. See [Configuration](docs/CONFIGURATION.md).

## Methodological notes

Cox losses use `2/n` normalization and Breslow ties. Pathway penalties use a differentiable Group Lasso approximation with epsilon `1e-4`; coefficients need not be exactly zero. Strict refits require a raw gradient infinity norm of at most `1e-5`. Feature filtering, scaling and imputation are fitted within training partitions.

Bootstrap intervals condition on fitted predictions; repeated partitions reuse patients. The EAC–STAD analysis is an internal, post-hoc comparison. The implementation is intended for research; clinical use requires independent validation.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data preparation](docs/DATA.md)
- [Configuration](docs/CONFIGURATION.md)
- [Training and evaluation](docs/REPRODUCIBILITY.md)
- [Verification scope](docs/VERIFICATION.md)
- [Source export and GitHub](docs/GITHUB.md)
- [Citation and licensing](docs/CITATION_AND_LICENSE.md)
