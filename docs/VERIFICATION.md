# Verification scope

Run the checks from the source checkout:

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/verify_code.py
```

The synthetic suite covers:

- Numerical-definition fingerprints across the core objective and solver modules.
- Thirty-six derivative and equivalence cases, and nine complete optimization cases.
- Zero-coupling equivalence to independent fitting and preprocessing isolation from held-out observations.
- CLI execution from another working directory and spawn-based multiprocessing.
- Protocol compatibility before cohort processing or training output.
- Source correspondence checks and exclusion of research assets from source exports.
- Workflow imports without loading research cohorts.

`verify_code.py` checks recorded source hashes and syntax-tree fingerprints. A passing check confirms the tested implementation properties; cohort-specific validation is performed separately by the experiment validators. Numerical tests do not establish clinical predictive performance.

The GitHub Actions workflow runs synthetic tests, source checks, command discovery and source packaging. It does not download research cohorts or initiate patient-cohort training.
