"""Keep source code, datasets and experiment artifacts in sibling directories."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("GASTRIC_MTL_WORKSPACE", str(ROOT.parent))).expanduser().resolve()
DATASETS = WORKSPACE / "datasets"
RESULTS = WORKSPACE / "results"
RAW_DATA = DATASETS / "raw"
PATHWAYS = DATASETS / "pathways"
PROCESSED = DATASETS / "processed"
GDC_METADATA = DATASETS / "metadata/gdc_expression_file_metadata.json"
COHORT_CACHE = PROCESSED / "cohorts.pkl"
THREE_CANCER = RESULTS / "three_cancer"
EAC_STAD = RESULTS / "eac_stad"
REFERENCE_INITIAL = RESULTS / "references/initial"
REFERENCE_SHARING = RESULTS / "references/sharing"
FIGURES = RESULTS / "figures"


def verify_recorded_source(relative_path, expected_sha256):
    """Check an exact source or an explicitly recorded relocation correspondence.

    A relocation match checks both the recorded source digest and the reviewed
    current-file digest. It records correspondence without claiming byte identity
    between the two versions.
    """
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source paths must be relative to the project root")
    path = ROOT / relative
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256:
        return "exact_current_source"
    record = json.loads((ROOT / "provenance/source_mapping.json").read_text())
    if record["original_source_hashes"].get(relative_path) != expected_sha256:
        raise ValueError("Unknown recorded source digest: " + relative_path)
    current = record["mapping"].get(relative_path)
    if not current:
        raise ValueError("No maintained source mapping: " + relative_path)
    path = ROOT / current
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["current_source_hashes"][current]:
        raise ValueError("Maintained source changed after correspondence audit: " + current)
    return "reviewed_relocation_not_byte_identity"
