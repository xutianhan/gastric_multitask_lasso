# Data preparation

Research inputs are managed outside the source repository. The cohort builder expects TCGA RNA-seq expression and clinical XML for ESCA, STAD and COAD, GDC file metadata, and pathway definitions.

## Input layout

The configured dataset root contains:

```text
raw/TCGA-{ESCA,STAD,COAD}/
  expression/expression_matrix.csv
  expression/<GDC-file-UUID>.tsv
  clinical/*.tsv
  clinical/clinical_survival.csv
metadata/gdc_expression_file_metadata.json
pathways/
  h.all.v2023.2.Hs.symbols.gmt
  c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt
processed/cohorts.pkl
```

The clinical `*.tsv` inputs contain XML. The survival CSV is an endpoint audit comparator; the final endpoint is derived from the XML patient and follow-up records. Confirmed deaths take precedence over earlier alive records; living patients use the latest documented follow-up.

GDC metadata must contain the complete response under `data.hits` and its `data.pagination.total` count. The loader reconciles expression matrix columns with complete TPM vectors from STAR files and then selects primary-tumour samples using the file metadata. It checks mapping coverage and patient alignment before constructing cohorts.

## Aligned cohort representation

The version-2 cache contains expression as patients by genes, gene identifiers, aligned patient identifiers, survival times in days, event indicators, clinical covariates and sample provenance. The EAC loader selects the exact histology label `Esophagus Adenocarcinoma, NOS` from ESCA.

The cache builder performs patient and endpoint alignment. Variance filtering, feature selection, imputation and standardization are learned within training partitions. Pathway groups use first-match assignment in a fixed GMT file order.

Build the cache with:

```bash
python scripts/run_workflow.py prepare-data
```

Use `--rebuild` to regenerate it from source inputs. Pickle inputs should come from trusted sources.

## Sources

Obtain TCGA inputs through the [GDC Data Portal](https://portal.gdc.cancer.gov/) and pathway definitions through [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/). Their access and redistribution terms apply separately from this software. Source packaging includes code and technical documentation.
