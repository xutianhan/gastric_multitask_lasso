# Configuration

Configure a workspace containing sibling `code/`, `datasets/` and `results/` directories. The default workspace is the parent of the source checkout. Data and generated research outputs stay outside the Git repository.

Set the workspace for one command:

```bash
python scripts/run_workflow.py --workspace /path/to/workspace train-three --help
```

Or configure it for a shell session:

```bash
export GASTRIC_MTL_WORKSPACE=/path/to/workspace
python scripts/run_workflow.py train-three --help
```

A command-line workspace takes precedence over the environment variable. The setting is resolved before workflow imports. Place `--workspace` before the workflow name.

`src/gastric_mtl/paths.py` defines the data, pathway, processed-input, study, reference and figure roots. Workflow-specific output names are generated beneath those roots. No machine-specific locations need to be edited in the experiment modules.

Use a separate workspace for a new protocol. Existing run protocols must agree with source and input hashes before training resumes. Run without Python's `-O` flag so scientific assertions remain enabled.
