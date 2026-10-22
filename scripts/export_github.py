#!/usr/bin/env python3
"""Build a source-only ZIP using an explicit allowlist; never upload anything."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ["README.md", "pyproject.toml", "requirements.txt", "environment.yml", ".gitignore", ".gitattributes"]
SOURCE_DIRS = ["src", "workflows", "scripts", "tests"]


def selected_files(root):
    files = [root / name for name in TOP_LEVEL]
    for directory in SOURCE_DIRS:
        files.extend((root / directory).rglob("*.py"))
    files.extend((root / "docs").rglob("*.md"))
    files.extend((root / ".github/workflows").glob("*.yml"))
    files.extend((root / "provenance").glob("*.json"))
    selected = sorted(set(files))
    for path in selected:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid export path: " + str(path))
        if path.stat().st_size > 2_000_000:
            raise ValueError("Unexpectedly large source file: " + str(path))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "results/exports/gastric-multitask-lasso-code.zip")
    parser.add_argument("--list", action="store_true", help="Print selected paths without creating an archive.")
    args = parser.parse_args()
    selected = selected_files(ROOT)
    if args.list:
        print("\n".join(str(path.relative_to(ROOT)) for path in selected))
        return
    output = args.output.expanduser().resolve()
    if output in [path.resolve() for path in selected]:
        parser.error("The archive cannot overwrite a source file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in selected}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("EXPORT_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "files": len(selected), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
