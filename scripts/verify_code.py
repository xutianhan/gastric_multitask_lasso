#!/usr/bin/env python3
"""Check current source correspondence without retaining duplicate source versions."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gastric_mtl.integrity import definition_fingerprints


def verify():
    expected = json.loads((ROOT / 'provenance/model_fingerprints.json').read_text())
    failures = []
    for name, definitions in expected.items():
        actual = definition_fingerprints(ROOT / name)
        if actual != definitions:
            failures.append(name)
    record = json.loads((ROOT / 'provenance/source_mapping.json').read_text())
    for name, expected_sha in record['current_source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected_sha:
            failures.append(name)
    return {'passed': not failures, 'numerical_modules': len(expected), 'failures': failures,
            'scope': 'Current source hashes and numerical-definition fingerprints.'}


if __name__ == '__main__':
    result = verify()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
