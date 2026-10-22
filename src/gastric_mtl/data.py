"""Patient-aligned, primary-tumour TCGA data for the 2026 submission analysis.

The cached expression CSV contains repeated patient column names and no sample
type. We map every column back to a local STAR file by its complete TPM vector,
then use the saved public GDC file metadata to identify primary tumours. No
outcome-dependent feature filtering, variance filtering, imputation, or scaling
is done here. Those operations belong inside model-training folds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from gastric_mtl.paths import RAW_DATA, PROCESSED, GDC_METADATA
OUT = PROCESSED
CTS = ("ESCA", "STAD", "COAD")
CACHE_VERSION = 2


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _text(node, name, recursive=True):
    seq = node.iter() if recursive else iter(node)
    found = [e.text.strip() for e in seq
             if _local(e.tag) == name and e.text and e.text.strip()]
    return found[0] if found else None


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _stage_number(value):
    if value is None:
        return np.nan
    match = re.fullmatch(r"(?:STAGE\s*)?(IV|III|II|I)(?:[ABC])?", value.upper().strip())
    return {"I": 1., "II": 2., "III": 3., "IV": 4.}.get(match.group(1), np.nan) if match else np.nan


def parse_clinical_xml(clinical_dir):
    """Read baseline covariates from XML saved with a .tsv extension.

    The patient and all nested follow-up records are examined. Any recorded
    death takes precedence over an earlier alive record; living patients use
    their longest documented follow-up. The old CSV is retained only for audit.
    """
    records = []
    errors = []
    for path in sorted(Path(clinical_dir).glob("*.tsv")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            errors.append({"file": path.name, "error": str(exc)})
            continue
        patient = next((e for e in root.iter() if _local(e.tag) == "patient"), None)
        if patient is None:
            errors.append({"file": path.name, "error": "No patient element"})
            continue
        pid = _text(patient, "bcr_patient_barcode", recursive=False)
        if not pid:
            errors.append({"file": path.name, "error": "No patient barcode"})
            continue
        stage_node = next((e for e in patient if _local(e.tag) == "stage_event"), None)
        stage = _text(stage_node, "pathologic_stage") if stage_node is not None else None
        baseline_pathology = next((e for e in patient if _local(e.tag) == "primary_pathology"), patient)
        age = _number(_text(baseline_pathology, "age_at_initial_pathologic_diagnosis"))
        age_source = "age_at_initial_pathologic_diagnosis"
        if not np.isfinite(age):
            birth_days = _number(_text(patient, "days_to_birth", recursive=False))
            if np.isfinite(birth_days) and birth_days < 0:
                age = -birth_days / 365.25
                age_source = "negative_days_to_birth_divided_by_365.25"
        sex = _text(patient, "gender", recursive=False)
        male = {"MALE": 1., "FEMALE": 0.}.get((sex or "").upper(), np.nan)
        # The XML may include baseline and subsequent follow-up records.
        deaths = [_number(e.text) for e in patient.iter() if _local(e.tag) == "days_to_death"]
        followup = [_number(e.text) for e in patient.iter() if _local(e.tag) == "days_to_last_followup"]
        deaths = [x for x in deaths if np.isfinite(x) and x >= 0]
        followup = [x for x in followup if np.isfinite(x) and x >= 0]
        vital = [e.text.strip().lower() for e in patient.iter()
                 if _local(e.tag) == "vital_status" and e.text and e.text.strip()]
        xml_event = 1 if (deaths or "dead" in vital or "deceased" in vital) else 0 if "alive" in vital else np.nan
        xml_time = max(deaths) if xml_event == 1 and deaths else max(followup) if xml_event == 0 and followup else np.nan
        records.append({
            "patient_id": pid, "age": age, "age_source": age_source,
            "sex": sex, "male": male, "stage": _stage_number(stage),
            "stage_label": stage,
            "histology": _text(patient, "histological_type"),
            "pathologic_T": _text(stage_node, "pathologic_T") if stage_node is not None else None,
            "pathologic_N": _text(stage_node, "pathologic_N") if stage_node is not None else None,
            "pathologic_M": _text(stage_node, "pathologic_M") if stage_node is not None else None,
            "xml_os_event": xml_event, "xml_os_time_days": xml_time,
            "xml_death_times": json.dumps(sorted(set(deaths))),
            "xml_followup_times": json.dumps(sorted(set(followup))),
            "xml_file": path.name,
        })
    df = pd.DataFrame(records).set_index("patient_id")
    duplicates = df.index[df.index.duplicated()].tolist()
    if duplicates:
        raise ValueError("Duplicate clinical XML patients require adjudication: " + repr(duplicates))
    return df, {"xml_records": len(df), "xml_parse_errors": errors, "duplicate_xml_patients": duplicates}


def _fingerprint(values):
    # Original STAR TPM values have four decimal places. The extra precision
    # preserves values while removing binary float serialisation differences.
    return hashlib.sha256(np.round(np.asarray(values, dtype=np.float64), 6).tobytes()).hexdigest()


def _file_metadata(metadata_path):
    source = json.loads(Path(metadata_path).read_text())
    hits = source["data"]["hits"]
    expected = source["data"]["pagination"]["total"]
    if len(hits) != expected:
        raise ValueError("GDC metadata cache is paginated/incomplete")
    result = {}
    for hit in hits:
        cases = hit.get("cases", [])
        if len(cases) != 1 or len(cases[0].get("samples", [])) != 1:
            raise ValueError("Ambiguous GDC case/sample association: " + hit["id"])
        case, sample = cases[0], cases[0]["samples"][0]
        result[hit["id"]] = {
            "file_id": hit["id"], "patient_id": case["submitter_id"],
            "project": case["project"]["project_id"],
            "sample_barcode": sample["submitter_id"],
            "sample_type": sample["sample_type"], "gdc_file_name": hit["file_name"],
        }
    return result


def _read_expression(ct, metadata):
    directory = RAW_DATA / ("TCGA-" + ct) / "expression"
    path = directory / "expression_matrix.csv"
    with open(path, newline="") as handle:
        headers = next(csv.reader(handle))[1:]
    # Pandas normally mangles duplicate headers with .1; retain the real header
    # independently and use positional columns until a unique patient is chosen.
    frame = pd.read_csv(path, index_col=0)
    if frame.index.duplicated().any():
        raise ValueError("Duplicate gene symbols in cached expression matrix")
    probe_genes = frame.index
    by_fp = defaultdict(list)
    for i in range(frame.shape[1]):
        by_fp[_fingerprint(frame.iloc[:, i].values)].append(i)
    columns = {}
    missing_files = []
    for raw in sorted(directory.glob("*.tsv")):
        uuid = raw.stem
        if uuid not in metadata:
            missing_files.append(uuid)
            continue
        values = pd.read_csv(raw, sep="\t", comment="#",
                             usecols=["gene_id", "gene_name", "tpm_unstranded"])
        values = values[values.gene_id.str.startswith("ENSG", na=False)]
        values = values.drop_duplicates("gene_name").set_index("gene_name")
        probe = values.reindex(probe_genes)["tpm_unstranded"].values
        matches = by_fp.get(_fingerprint(probe), [])
        if len(matches) != 1:
            raise ValueError(f"Expression fingerprint is not unique for {uuid}: {matches}")
        column = matches[0]
        if column in columns:
            raise ValueError("Multiple raw files match matrix column " + str(column))
        record = dict(metadata[uuid])
        if headers[column] != record["patient_id"]:
            raise ValueError("Expression header disagrees with GDC patient association")
        if record["project"] != "TCGA-" + ct:
            raise ValueError("GDC project mismatch")
        record["matrix_column"] = column
        record["source_annotated_genes"] = len(values)
        record["missing_or_invalid_tpm_genes"] = int((~np.isfinite(probe) | (probe < 0)).sum())
        record["complete_valid_expression"] = record["missing_or_invalid_tpm_genes"] == 0
        columns[column] = record
    if missing_files or len(columns) != len(headers):
        raise ValueError(f"Incomplete provenance mapping: {ct}, unknown files={missing_files}, mapped={len(columns)}/{len(headers)}")
    manifest = pd.DataFrame([columns[i] for i in range(len(headers))])
    return frame, manifest, path


def build_submission_data(output_dir=OUT, metadata_path=None):
    output_dir = Path(output_dir)
    work = output_dir
    work.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_path or GDC_METADATA)
    metadata = _file_metadata(metadata_path)
    audit = {
        "cache_version": CACHE_VERSION,
        "created_date": "2026-09-14",
        "metadata_source": "https://api.gdc.cancer.gov/files",
        "metadata_retrieved_date": "2026-09-14",
        "metadata_sha256": _sha256(metadata_path),
        "expression_unit": "TPM (STAR tpm_unstranded); log2(TPM + 1)",
        "time_unit": "days",
        "endpoint_source": "Local clinical XML: confirmed death at any visit takes precedence, with recorded death day; otherwise latest nonnegative documented follow-up for confirmed alive patients. Confirmed death without a death day remains missing and is excluded.",
        "legacy_endpoint_source": "clinical_survival.csv retained as audit comparator only",
        "primary_tumour_rule": "GDC sample_type exactly Primary Tumor; exclude files with incomplete or invalid TPM gene vectors",
        "replicate_rule": "One per patient: sample barcode then file UUID in ascending lexical order; no expression or outcome criterion",
        "survival_exclusions": "Missing/nonfinite OS time or event, nonbinary event, or OS time <= 0",
        "gene_rule": "Sorted intersection of original gene symbol annotations across three cohorts; no variance or outcome filtering",
        "fold_operations_required": ["variance filtering", "feature selection", "missing-value imputation if needed", "standardisation", "hyperparameter selection"],
        "cohorts": {},
    }
    prepared = {}
    for ct in CTS:
        print("Rebuilding", ct, flush=True)
        expr, manifest, expression_path = _read_expression(ct, metadata)
        manifest.to_csv(output_dir / f"sample_manifest_{ct}.csv", index=False)
        clinical_dir = RAW_DATA / ("TCGA-" + ct) / "clinical"
        clinical_path = clinical_dir / "clinical_survival.csv"
        survival = pd.read_csv(clinical_path, index_col=0)
        if survival.index.duplicated().any():
            raise ValueError("Duplicate patient IDs in survival source")
        covariates, xml_audit = parse_clinical_xml(clinical_dir)
        all_primary = manifest[manifest.sample_type.eq("Primary Tumor")]
        primary = all_primary[all_primary.complete_valid_expression].sort_values(["sample_barcode", "file_id"])
        unique_primary = primary.drop_duplicates("patient_id", keep="first").set_index("patient_id")
        candidate_ids = sorted(set(unique_primary.index).intersection(covariates.index))
        endpoint = covariates.loc[candidate_ids].copy()
        time = pd.to_numeric(endpoint["xml_os_time_days"], errors="coerce")
        event = pd.to_numeric(endpoint["xml_os_event"], errors="coerce")
        legacy_time = pd.to_numeric(survival.reindex(candidate_ids)["OS.time"], errors="coerce")
        legacy_event = pd.to_numeric(survival.reindex(candidate_ids)["OS"], errors="coerce")
        eligible = np.isfinite(time) & time.gt(0) & event.isin([0, 1])
        legacy_eligible = np.isfinite(legacy_time) & legacy_time.gt(0) & legacy_event.isin([0, 1])
        endpoint_audit = pd.DataFrame({"xml_time_days": time, "xml_event": event,
                                      "csv_time_days": legacy_time, "csv_event": legacy_event,
                                      "xml_eligible": eligible, "csv_eligible": legacy_eligible})
        endpoint_audit.to_csv(output_dir / f"endpoint_source_audit_{ct}.csv", index_label="patient_id")
        pids = endpoint.index[eligible].tolist()
        selected = unique_primary.loc[pids].copy()
        selected.to_csv(output_dir / f"selected_primary_samples_{ct}.csv")
        clinic = covariates.reindex(pids).copy()
        clinic["time"] = time.loc[pids].to_numpy(float)
        clinic["event"] = event.loc[pids].to_numpy(int)
        clinic["csv_os_time_days"] = legacy_time.loc[pids].to_numpy(float)
        clinic["csv_os_event"] = legacy_event.loc[pids].to_numpy(float)
        clinic.to_csv(output_dir / f"clinical_covariates_{ct}.csv", index_label="patient_id")
        # Keep every endpoint correction and restored record reviewable.
        discrep = clinic.loc[clinic.csv_os_time_days.isna() |
                            (clinic.time - clinic.csv_os_time_days).abs().gt(0.5) |
                            clinic.event.ne(clinic.csv_os_event)]
        discrep.to_csv(output_dir / f"endpoint_discrepancies_{ct}.csv", index_label="patient_id")
        matrix = expr.iloc[:, selected.matrix_column.to_numpy(int)].to_numpy(dtype=np.float32).T.copy()
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("TPM source has invalid values; explicit handling required")
        matrix = np.log2(matrix + np.float32(1.0))
        counts = Counter(manifest.patient_id)
        audit["cohorts"][ct] = {
            "raw_expression_genes": len(expr), "raw_expression_columns": len(manifest),
            "raw_unique_expression_patients": len(counts),
            "raw_duplicate_patient_columns": sum(v - 1 for v in counts.values()),
            "raw_sample_types": manifest.sample_type.value_counts().to_dict(),
            "primary_tumour_files": len(all_primary), "complete_primary_tumour_files": len(primary),
            "incomplete_or_invalid_primary_files_removed": len(all_primary) - len(primary),
            "incomplete_expression_files": manifest.loc[~manifest.complete_valid_expression, ["file_id", "patient_id", "sample_type", "source_annotated_genes", "missing_or_invalid_tpm_genes"]].to_dict("records"),
            "unique_primary_tumour_patients": len(unique_primary),
            "primary_replicates_removed": len(primary) - len(unique_primary),
            "clinical_patients": len(survival), "primary_without_clinical": len(unique_primary) - len(candidate_ids),
            "missing_or_invalid_time_or_event": int((~(np.isfinite(time) & event.isin([0, 1]))).sum()),
            "nonpositive_time": int((np.isfinite(time) & time.le(0)).sum()),
            "analysis_patients": len(pids), "events": int(clinic.event.sum()),
            "censored": int(len(pids) - clinic.event.sum()),
            "median_observed_time_days": float(clinic.time.median()),
            "max_observed_time_days": float(clinic.time.max()),
            "age_available": int(clinic.age.notna().sum()),
            "sex_available": int(clinic.male.notna().sum()),
            "stage_available": int(clinic.stage.notna().sum()),
            "clinical_complete_cases": int(clinic[["age", "male", "stage"]].notna().all(axis=1).sum()),
            "stage_counts": {str(k): int(v) for k, v in clinic.stage.value_counts(dropna=False).items()},
            "histology_counts": {str(k): int(v) for k, v in clinic.histology.value_counts(dropna=False).items()},
            "xml_endpoint_discrepancies": len(discrep),
            "patients_restored_by_xml_endpoint": endpoint.index[eligible & ~legacy_eligible].tolist(),
            "patients_excluded_by_xml_endpoint": endpoint.index[~eligible & legacy_eligible].tolist(),
            "corrected_event_status_patients": clinic.index[clinic.event.ne(clinic.csv_os_event)].tolist(),
            "expression_sha256": _sha256(expression_path),
            "clinical_sha256": _sha256(clinical_path),
            **xml_audit,
        }
        prepared[ct] = {"X_log2": matrix, "genes": expr.index.to_numpy(str),
                        "patient_ids": np.asarray(pids), "time": clinic.time.to_numpy(float),
                        "event": clinic.event.to_numpy(int), "clinical": clinic,
                        "provenance": selected}
        print(ct, len(pids), "patients", int(clinic.event.sum()), "events", flush=True)
    common = np.asarray(sorted(set.intersection(*(set(prepared[ct]["genes"]) for ct in CTS))))
    for ct in CTS:
        index = pd.Index(prepared[ct]["genes"]).get_indexer(common)
        prepared[ct]["X_log2"] = prepared[ct]["X_log2"][:, index]
        prepared[ct]["genes"] = common.copy()
        audit["cohorts"][ct]["common_annotated_genes"] = len(common)
    audit["common_annotated_genes"] = len(common)
    with open(work / "cohorts.pkl", "wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "cohorts": prepared}, handle, protocol=4)
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return prepared


def load_submission_cohorts(output_dir=OUT, rebuild=False):
    """Return {cancer: {X_log2, genes, patient_ids, time, event, clinical, provenance}}.

    X_log2 has shape patients x genes; all arrays and clinical DataFrame rows are
    in identical patient-ID order. Clinical numeric predictors are age, male,
    stage; missing values remain NaN for training-fold imputation. Stage is the
    baseline pathological stage (I=1 to IV=4). Survival time is in days.
    """
    cache = Path(output_dir) / "cohorts.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as handle:
            loaded = pickle.load(handle)
        if loaded["version"] != CACHE_VERSION:
            raise ValueError("Rebuild outdated submission cohort cache")
        return loaded["cohorts"]
    return build_submission_data(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    load_submission_cohorts(rebuild=args.rebuild)
