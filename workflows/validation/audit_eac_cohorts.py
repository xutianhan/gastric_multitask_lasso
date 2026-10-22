#!/usr/bin/env python3
"""Independently re-read source clinical XML and report cohort characteristics."""
from gastric_mtl.paths import RAW_DATA, WORKSPACE
from workflows.experiments.eac_stad import *
from gastric_mtl.data import parse_clinical_xml


def main():
    co=load_submission_cohorts();stats={};sources={}
    for c in CTS:
        source='ESCA' if c=='EAC' else c
        folder=RAW_DATA/('TCGA-'+source)/'clinical'
        clinical,audit=parse_clinical_xml(folder)
        cl=co[c]['clinical'];ids=co[c]['patient_ids'].astype(str)
        raw=clinical.loc[ids]
        assert raw['histology'].tolist()==cl['histology'].tolist()
        assert raw['sex'].tolist()==cl['sex'].tolist()
        for key in ['age','stage']:
            np.testing.assert_allclose(raw[key].to_numpy(float),cl[key].to_numpy(float),atol=0,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(raw['xml_os_time_days'].to_numpy(float),co[c]['time'])
        np.testing.assert_array_equal(raw['xml_os_event'].to_numpy(float),co[c]['event'])
        age=cl['age'].to_numpy(float);stage=cl['stage'].to_numpy(float)
        t=co[c]['time'];e=co[c]['event']
        stats[c]={'n':len(t),'deaths':int(e.sum()),'age_quartiles':np.nanquantile(age,[.25,.5,.75]).tolist(),
            'age_missing':int(np.isnan(age).sum()),'male':int((cl['sex']=='MALE').sum()),
            'stage_counts':{str(i):int((stage==i).sum()) for i in range(1,5)},
            'stage_missing':int(np.isnan(stage).sum()),'observed_time_quartiles':np.quantile(t,[.25,.5,.75]).tolist(),
            'observed_beyond_1y':int((t>365.25).sum()),'observed_beyond_3y':int((t>1095.75).sum()),
            'histology_counts':{str(k):int(v) for k,v in cl['histology'].value_counts().items()}}
        sources.update({str(p.relative_to(WORKSPACE)):digest(p) for p in folder.glob('*.tsv')})
    manifest=json.loads((OUT.parent/'split_manifest.json').read_text())
    outer=[r for r in manifest if r['phase']=='outer'];inner=[r for r in manifest if r['phase']!='outer']
    v={'passed':True,'raw_clinical_xml_reconstruction_matches_cached_cohorts':True,'stats':stats,'source_xml_sha256':sources,
        'outer_feature_range':[min(v['features'] for v in outer),max(v['features'] for v in outer)],
        'inner_feature_range':[min(v['features'] for v in inner),max(v['features'] for v in inner)],
        'groups':sorted(set(v['groups'] for v in manifest)),
        'cohort_csv_sha256':{c:digest(OUT.parent/f'cohort_{c}.csv') for c in CTS}}
    dump(OUT.parent/'cohort_characteristics.json',v)
    print(json.dumps({k:v for k,v in v.items() if k!='source_xml_sha256'},indent=2))

if __name__=='__main__':main()
