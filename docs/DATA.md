# Data

No data file is part of this repository. This document lists every input, where it comes from and where the code expects it. Sizes and SHA-256 checksums are in `docs/INPUT_FILES.md`, which is generated from the records.

## 1. The data root

Every input is read from one folder, the data root. It is set by the environment variable `IGA_STORAGE_ROOT` or, when the variable is unset, by the first line of `storage_root.txt` at the repository root. The code stops with an error when neither is set.

Recorded paths are written as `$IGA_STORAGE_ROOT/<relative path>`. The relative paths, including the folder names `IGA_MAIN`, `IGA_GENEFORMER_EXT` and `MADA`, are the ones that the code and the records of this copy use. Keep them when you build your own data root. The first two replace the names under which the analysis ran (see the README, under "Layout").

## 2. Public data sets and downloads

| Input | Reference | Files used | Place under the data root |
|---|---|---|---|
| Virtual Cell Challenge 2025, H1 cells | Arc Institute, Arc Virtual Cell Atlas, `virtual-cell-challenge` (GitHub: `ArcInstitute/arc-virtual-cell-atlas`) | `adata_Training.h5ad`, `adata_Validation.h5ad`, `adata_Test.h5ad` | `IGA_MAIN/data/vcc/{train,validation,test}/` |
| Perturb-seq screens, RPE1 and K562 essential genes | Replogle and Weissman (2022), Figshare+, doi:10.25452/figshare.plus.20029387.v1 (CC BY 4.0) | `rpe1_raw_singlecell_01.h5ad`, `K562_essential_raw_singlecell_01.h5ad` (raw single-cell files; normalisation is done by this code) and the two `*_normalized_bulk_01.h5ad` files | `IGA_MAIN/data/replogle/raw_singlecell/` and `IGA_MAIN/data/replogle/raw/` |
| DepMap 24Q4 Public | Broad DepMap (2024), Figshare+, doi:10.25452/figshare.plus.27993248.v1 (CC BY 4.0) | `CRISPRGeneEffect.csv`, `Model.csv` | `bioinf2027/downloads/depmap_24Q4/` |
| GenePT gene vectors, version 2 | Chen (2024), Zenodo, doi:10.5281/zenodo.10833191 (CC BY 4.0) | `GenePT_emebdding_v2.zip`; the vectors used are `GenePT_gene_embedding_ada_text.pickle` | `bioinf2027/downloads/genept_v2/` |
| Geneformer V2-104M | Hugging Face, `ctheodoris/Geneformer` | `Geneformer-V2-104M/model.safetensors`, `config.json`, and the token and gene dictionaries of the repository | `MADA/cache/foundation_models/geneformer-repo/` |
| scGPT, whole-human model | scGPT model release (see `bioinf2027/results/E2/downloads.json` for the source of each file) | `best_model.pt`, `vocab.json`, `args.json` | `bioinf2027/downloads/scgpt_whole_human/` |
| Stack-Large | Hugging Face, `arcinstitute/Stack-Large` (non-commercial licence, see its `MODEL_LICENSE.md`) | `bc_large.ckpt`, `basecount_1000per_15000max.pkl` | `bioinf2027/downloads/stack_large/` |
| scBaseCount sample metadata, two releases | see `bioinf2027/results/E2/downloads.json` | `sample_metadata.parquet` | `bioinf2027/downloads/scbasecount_*/` |
| HGNC complete set | HGNC (see `bioinf2027/results/E5/supplement_2/` for the download record) | `hgnc_complete_set.txt` | `bioinf2027/downloads/hgnc/` |

The download scripts are `make_e2_downloads.py`, `make_e5_depmap_download.py` and `make_e5_hgnc_download.py`. Each writes a record with the source address, the size and the checksum of every file. The single-cell files of the Virtual Cell Challenge and of the Replogle collection are downloaded by hand from the addresses above.

Check the terms of each source before you redistribute any file. The Stack weights are for non-commercial use.

## 3. Intermediate files from the first version of this work

The analysis started from intermediate files that the code of the first version of this work had built from the same raw data. This repository reads them and does not build them:

- the list of 2,000 variable genes and the gene names of the Virtual Cell Challenge matrix;
- the stored all-cells score of each Virtual Cell Challenge group and the stored expression-shift features;
- the Geneformer cell embeddings of the Virtual Cell Challenge groups and of the constructs of the two external screens;
- the per-cell expression-shift arrays of the two external screens;
- the eligibility lists and the first score files of the two external screens.

`docs/INPUT_FILES.md`, section 2, lists these files with their sizes and checksums. It also lists the folders of the first version that are read as a whole (the per-batch files of the Virtual Cell Challenge expression-shift features and the per-cell Geneformer embeddings), with the number of files and the checksum of the file listing.

What this repository does about them:

- It rebuilds the all-cells score and the fixed-count scores from the raw single-cell files (`bioinf2027/src/bioinf2027/endpoint.py`, `make_e1a_labels.py`) and compares the rebuilt scores with the stored ones (`make_e1a_r0.py`, result in `bioinf2027/endpoints/builder_checks.json`).
- It embeds cells with Geneformer again for the target-token analysis and compares unchanged inputs with the stored embeddings (`make_d2_embeddings.py`).

A rerun from raw data alone therefore needs the first version's code for the five items above, or these files from the authors. The folder `first_version/` holds that code for 23 of the 34 files; its README states what it built, what it cannot rebuild, and the result of a rebuild of the 11 files that need no GPU.

`bioinf2027/scripts/make_d0_checks.py`, which compares with the earlier manuscript version, reads ten further files of the first version (saved result tables, one log and a second copy of the Geneformer token dictionary). Section 2 does not list them and they are not distributed.

## 4. Files that are not distributed

Files with one row per Virtual Cell Challenge group or gene, and most files with one row per construct or gene of the external screens, are left out. Each is written again by the script named here when the data root is in place.

| File | Written by |
|---|---|
| `bioinf2027/endpoints/null_labels.csv`, `builder_check_bc_rows.csv`, `builder_check_a_constructs.csv`, `R0_control_mean_spread.csv` | `make_e1a_r0.py` |
| `bioinf2027/results/D2/manipulation.csv`, `delta_per_flex_run.csv`, `supplement/delta_batch.csv` | `make_d2_aggregate.py` (first two), `make_d2_supplement.py` |
| `bioinf2027/results/E5/endpoint_b/constructs_*.csv`, `genes_*.csv`, `vcc_genes.csv` | `make_e5_endpoint_b.py`, `make_e5_endpoint_b_vcc.py` |
| `bioinf2027/diag/D1_target_in_features_external_per_construct.csv`, `D1b_b_external_per_construct.csv`, `E4_external_z.csv` | `make_d1_external.py`, `make_d1b.py`, `diagnose_e4_external_z.py` |

Three files with one row per construct or gene are included because table scripts list them as inputs: `bioinf2027/results/S1_supplement/external_labels.csv`, `bioinf2027/results/E5/supplement/pool_constructs.csv` and `bioinf2027/results/E5/supplement/depmap_mapping.csv`. They derive from the Replogle and DepMap data sets, both released under CC BY 4.0.
