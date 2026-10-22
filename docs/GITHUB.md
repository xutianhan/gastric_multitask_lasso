# Source export and GitHub

Initialize Git inside the source checkout. Keep research inputs and generated outputs in the configured external workspace.

```bash
python -m unittest discover -s tests -v
python scripts/verify_code.py
python scripts/export_github.py --list
python scripts/export_github.py

git init
git add .
git status --short
git diff --cached --stat
git commit -m "Publish multitask Cox research implementation"
```

Create a GitHub repository and follow its remote and push instructions. Alternatively, use the ZIP created by `export_github.py`. Its output location is printed after export and can be set with `--output /path/to/source.zip`.

The exporter uses an explicit allowlist for Python source, tests, technical documentation, package configuration, CI and source-integrity records. It adds an SHA-256 manifest so the exported files can be verified. The script packages files locally and does not upload them.

See [Citation and licensing](CITATION_AND_LICENSE.md) for the software's licensing status.
