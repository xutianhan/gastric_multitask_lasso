# Training and evaluation

Run commands from the source checkout using a configured workspace and the input schema in [Data preparation](DATA.md). The workflow runner lists available stages:

```bash
python scripts/run_workflow.py list
```

## Software checks

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/verify_code.py
```

The reference environment is Python 3.9.7, NumPy 1.19.2, SciPy 1.6.0, pandas 1.2.1, scikit-learn 0.24.2 and Matplotlib 3.3.4. The dependency check reports both runtime versions and installed distribution metadata. Use a clean environment for a new study.

## Study stages and prerequisites

| Stage | Inputs and purpose |
| --- | --- |
| `prepare-data` | Builds the aligned cohort cache from source expression, clinical and GDC metadata inputs |
| `initial-evaluation` | Generates single-task references and patient partitions used by strict comparisons |
| `train-three` | Uses the cache, pathways and reference partitions to fit independent, pairwise and three-task candidates |
| `preflight-eac` | Checks the histology subset and reference partitions, then freezes the EAC–STAD protocol |
| `train-eac` | Fits independent and paired EAC–STAD candidates |
| `audit-eac` | Reconciles cohort characteristics with source clinical XML |
| `validate-three`, `validate-eac` | Use completed fit records, selection records and the numerical solver validation report to recompute objectives and predictions |
| `summarize-three` | Aggregates validated folds and compares selections with the study's reference-sharing experiment |
| `summarize-eac` | Aggregates validated folds, conditional intervals and selection frequencies |
| `figures-three`, `figures-eac` | Render from the corresponding validated summary records |

Stages consume study-specific inputs and preceding outputs. In particular, `summarize-three` also consumes reference-sharing predictions and selections, and `validate-eac` compares the STAD clinical reference with the three-cancer experiment. These comparison stages are intended for workspaces configured with those reference records; they are not independent training commands.

## Reference partitions

In a new workspace, generate the two reference partition sets with:

```bash
python scripts/run_workflow.py initial-evaluation --seed 42 --name nested_primary
python scripts/run_workflow.py initial-evaluation --seed 2026 --name nested_repeat_seed2026
```

This stage uses the reference optimizer. The strict comparison stages apply the final convergence criterion. The study uses five outer folds and three inner folds for each seed.

## Model fitting

```bash
python scripts/run_workflow.py train-three --workers 4
python scripts/run_workflow.py preflight-eac
python scripts/run_workflow.py train-eac --workers 4
```

Use `--seed 42 --fold 0` to run one outer-fold job. A full invocation writes overall completion only after all ten outer-fold jobs are complete. Current source and input hashes are frozen before training. Resume only under an identical protocol; use a fresh workspace for protocol changes.

## Evaluation and reporting

Once the relevant validation prerequisites in the table are satisfied:

```bash
python scripts/run_workflow.py validate-three
python scripts/run_workflow.py validate-eac
python scripts/run_workflow.py summarize-three
python scripts/run_workflow.py summarize-eac
python scripts/run_workflow.py figures-three
python scripts/run_workflow.py figures-eac
```

Validation checks held-out predictions, objective values, convergence, patient partitions and selection records. Summary workflows compute concordance, one- and three-year IPCW Brier scores, paired differences, sharing-strength curves and selection frequencies. Figure rendering does not edit manuscript documents.

## Interpretation

The EAC–STAD comparison is post-hoc and internally validated; STAD is not restricted to the CIN subtype. Bootstrap intervals condition on fitted predictions and training-derived censoring estimates, omitting refitting and selection uncertainty. Repeated partitions reuse patients. Clinical molecular-score cross-fitting is not fully nested with respect to candidate selection. These properties should be retained when reporting results.
