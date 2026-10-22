"""Histology-defined EAC/STAD cohorts and training-only preprocessing.

The parent archive and all original three-cancer results are read-only inputs.
This module does not alter original global cancer lists or cached cohort files.
"""
import numpy as np
from .data import load_submission_cohorts

CTS = ['EAC', 'STAD']
EAC_LABEL = 'Esophagus Adenocarcinoma, NOS'
ESCC_LABEL = 'Esophagus Squamous Cell Carcinoma'


def load_eac_cohorts():
    original = load_submission_cohorts()
    parent = original['ESCA']
    ids = np.asarray(parent['patient_ids']).astype(str)
    assert np.array_equal(parent['clinical'].index.astype(str), ids)
    hist = parent['clinical']['histology'].astype(str).to_numpy()
    assert set(hist) == {EAC_LABEL, ESCC_LABEL}
    keep = np.flatnonzero(hist == EAC_LABEL)
    assert len(keep) == 89 and int(parent['event'][keep].sum()) == 45
    eac = dict(parent)
    for key in ['X_log2', 'patient_ids', 'time', 'event']:
        eac[key] = parent[key][keep].copy()
    eac['clinical'] = parent['clinical'].iloc[keep].copy()
    eac['parent_indices'] = keep
    eac['histology_definition'] = EAC_LABEL
    co = {'EAC': eac, 'STAD': original['STAD']}
    assert np.array_equal(eac['genes'], co['STAD']['genes'])
    assert len(co['STAD']['patient_ids']) == 380 and int(co['STAD']['event'].sum()) == 155
    assert not set(eac['patient_ids']) & set(co['STAD']['patient_ids'])
    for c in CTS:
        assert np.array_equal(co[c]['clinical'].index.astype(str), co[c]['patient_ids'].astype(str))
    return co


def prepare(cohorts, tr, te, labels):
    """Refit common gene variance filters and each task's scaling in training."""
    assert list(cohorts) == CTS and len(tr) == len(te) == len(CTS)
    base = labels < labels.max()  # The last label is the unassigned gene group.
    sd = [cohorts[c]['X_log2'][tr[k]].std(axis=0) for k, c in enumerate(CTS)]
    mask = base & np.logical_and.reduce([s > 1e-8 for s in sd])
    idx = np.flatnonzero(mask)
    tasks, tests, scalers = [], [], []
    for k, c in enumerate(CTS):
        assert not set(tr[k]) & set(te[k])
        raw = cohorts[c]['X_log2'][:, idx].astype(float)
        mean = raw[tr[k]].mean(axis=0)
        scale = raw[tr[k]].std(axis=0)
        assert np.all(scale > 1e-8)
        tasks.append({'X': np.ascontiguousarray((raw[tr[k]]-mean)/scale),
                      'time': cohorts[c]['time'][tr[k]], 'event': cohorts[c]['event'][tr[k]]})
        tests.append({'X': np.ascontiguousarray((raw[te[k]]-mean)/scale),
                      'time': cohorts[c]['time'][te[k]], 'event': cohorts[c]['event'][te[k]]})
        scalers.append((mean, scale))
    return tasks, tests, labels[idx], idx, scalers
