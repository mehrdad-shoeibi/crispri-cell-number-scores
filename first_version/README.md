# Code of the first version of this work

These 21 files are copied from the authors' working repository. The only changes are the name replacements that the top-level README describes: three folder names, the Python constant `IGA_MAIN`, one project name in the docstring of `iga_paths.py` and one word in four docstrings. They built 23 of the 34 intermediate files that `docs/INPUT_FILES.md`, section 2, lists, before the analysis of the article began. They are kept here as a record of how those files were made.

**Status.** Apart from those names, the code is given as it ran. The other comments and docstrings are those of the earlier work. They name working sessions and notes that are not distributed. The steps that need no GPU were run again twice from the raw downloads, once before and once after the names were replaced (see "Rebuild check"). The GPU steps were not rerun. The tests of this repository that compare with `x_cell_accumulator.py` look for it under `earlier_revision/pipeline/` at the top of the repository, not under `first_version/`, so they are skipped in this copy.

## What each script wrote

Paths are relative to the data root (`docs/DATA.md`, section 1).

| Script | Files it wrote | What it reads | GPU |
|---|---|---|---|
| `src/data/vcc/loader.py`, function `load_vcc_pseudobulk` (with `hvg.py`, `schema.py`, `delta_features.py`, `gene_selection.py`) | `IGA_MAIN/cache/vcc/nt_baseline/batch_order.npy`, `IGA_MAIN/cache/vcc/nt_baseline/mean_log1p.npy`, `IGA_MAIN/cache/vcc/var_gene_names.npy`, `IGA_MAIN/cache/vcc/variants/delta_hvg/selected_genes.npy` and the per-batch files under `IGA_MAIN/cache/vcc/variants/delta_hvg/shards/`. It also writes per-batch control-cell arrays under `nt_baseline/per_batch_full_log1p/` and `variants/delta_hvg/meta.json`, which section 2 does not list | the three Virtual Cell Challenge files and their `pert_counts_*.csv` files | no |
| `scripts/v4/harmonized_endpoint_v1/step1_subsample_sizes.py` | `IGA_MAIN/results_v4/harmonized_endpoint_v1/subsample_sizes.json` | the per-batch files above and the three Virtual Cell Challenge files | no |
| `scripts/v4/harmonized_endpoint_v1/step3_eligibility.py` (argument `rpe1`) | `IGA_MAIN/results_v4/harmonized_endpoint_v1/coverage_summary.json` (entry `rpe1`), `rpe1_eligibility.csv` | the subsample sizes, the RPE1 pseudobulk file and the RPE1 single-cell file | no |
| `scripts/v4/harmonized_endpoint_v1/run_replogle_endpoint.py` (argument `rpe1`) | `IGA_MAIN/results_v4/harmonized_endpoint_v1/rpe1_endpoint_by_construct.csv` | the subsample sizes, the eligibility list, the RPE1 pseudobulk and single-cell files | no |
| `scripts/v4/harmonized_endpoint_k562_essential_v1/step_eligibility.py` | `IGA_MAIN/results_v4/harmonized_endpoint_k562_essential_v1/coverage_summary.json`, `eligibility.csv` | the subsample sizes, the K562-essential pseudobulk and single-cell files | no |
| `scripts/v4/harmonized_endpoint_k562_essential_v1/run_endpoint.py` | `IGA_MAIN/results_v4/harmonized_endpoint_k562_essential_v1/endpoint_by_construct.csv` | the subsample sizes, the eligibility list, the K562-essential pseudobulk and single-cell files | no |
| `earlier_revision/pipeline/phase2_reagg_and_oracle.py` | `IGA_GENEFORMER_EXT/scratch/nt_oidx.npy`, `scratch/mu_nt_emb.npy` and further scratch files for the next two scripts | per-cell Geneformer embeddings and feature files of an earlier feature pipeline (see below), and `IGA_GENEFORMER_EXT/manifests/rpe1_construct_sets.json` | no |
| `earlier_revision/pipeline/phase2_embed_and_xcell.py` (with `x_cell_accumulator.py`), run after the script above | `IGA_GENEFORMER_EXT/checkpoints/rpe1_xcell/xcell_state.pkl`, `IGA_GENEFORMER_EXT/x_cell/x_cell_rpe1_mapped.npz`, `x_cell_rpe1_native.npz`, `IGA_GENEFORMER_EXT/rpe1_topup/rpe1_43_percell_emb.npy`, `rpe1_43_percell_index.parquet` | the RPE1 single-cell file, `arm1_rpe1_construct_list.csv`, `rpe1_construct_sets.json`, the two gene lists above, `scratch/mu_nt_emb.npy` and `scratch/nt_oidx.npy`, the Geneformer V2-104M weights and token dictionary | yes |
| `earlier_revision/pipeline/phase2_assemble_qc.py` | `IGA_GENEFORMER_EXT/rpe1_topup/rpe1_construct_index.parquet`, `rpe1_topup/z_rpe1_1932.npy`, `IGA_GENEFORMER_EXT/x_cell/x_cell_rpe1_1932_mapped.npy` | the outputs of the two scripts above, one file of the earlier feature pipeline (`phasec_rpe1_expression_features_full_v1.parquet`), the cell and gene tables of the RPE1 single-cell and pseudobulk files, `arm1_rpe1_construct_list.csv`, `rpe1_construct_sets.json` and the two gene lists | no |
| `earlier_revision/pipeline/phase3_runner.py` (`--mode prod`) | `IGA_GENEFORMER_EXT/k562_essential/checkpoints/xcell_state.pkl`, `k562_essential/x_cell/x_cell_k562_1903_mapped.npy`, `x_cell_k562_native.npz`, `k562_essential/z/k562_construct_index.parquet`, `z_k562_1903.npy` and the per-cell embedding units under `k562_essential/emb_units/` (474 files) | the K562-essential single-cell file, two construct lists, the two gene lists above, the Geneformer V2-104M weights and token dictionary | yes |

The stored `harmonized_endpoint_v1/coverage_summary.json` holds a second entry, `gwps`. The same script wrote it with the argument `gwps` from the K562 genome-wide screen, which is not among the downloads of this repository, and the K562-essential pseudobulk file. The code under `bioinf2027/` reads only the entry `rpe1`.

`iga_paths.py` resolves the data root from the environment variable `IGA_STORAGE_ROOT` or from a file `storage_root.txt` beside it. Each script looks for `iga_paths.py` in its parent folders, so the relative layout of this folder is the layout the scripts ran in.

## What this code does not rebuild

1. Seven of the 34 files have no writer in this folder:
   `IGA_GENEFORMER_EXT/manifests/phase2_rpe1_manifest.json`,
   `IGA_MAIN/results_v4/k562_essential_coverage_gate_v1/gemgroup_nt.csv`,
   `IGA_MAIN/results_v4/transfer_pretest_lock_v1/arm1_rpe1_construct_list.csv`,
   `MADA/cache/foundation_models/geneformer/full_feature_extraction_v2/cell_embedding_index.parquet`,
   `MADA/cache/foundation_models/geneformer/full_feature_extraction_v2/geneformer_row_features_full.parquet`,
   `MADA/data/metadata/vcc/final_labels.parquet` and
   `MADA_data/processed/phasec/rpe1_features/full/master_cell_index_v1.parquet`.
   The two files under `MADA/cache/` and the file under `MADA_data/` belong to an earlier feature pipeline of the authors. Its code is not part of this repository. The writers of the other four were not found in the working repository.
2. Four inputs of the scripts above have no writer here either: `IGA_GENEFORMER_EXT/manifests/rpe1_construct_sets.json`, `IGA_MAIN/results_v4/transfer_pretest_lock_v1/arm1_rpe1_construct_list.csv`, `IGA_MAIN/results_v4/k562_essential_coverage_gate_v1/k562_essential_eligibility.csv` and `IGA_MAIN/results_v4/arm2_pretest_lock_v1/evaluation_constructs.csv`.
3. The Geneformer embeddings of 1,889 of the 1,932 RPE1 constructs were aggregated from per-cell embeddings of the earlier feature pipeline (`phase2_reagg_and_oracle.py`). This folder embeds only the cells of the other 43 constructs. The same holds for the per-cell embedding folders that `docs/INPUT_FILES.md`, section 2, lists under `MADA/` and `MADA_data/`.
4. None of the 21 files only builds the Virtual Cell Challenge caches. `load_vcc_pseudobulk` writes the control baselines and the gene names on its first call, the gene selection and the training per-batch files on its first call with `feature_transform="delta"` and `gene_selection="hvg"`, and the validation and test per-batch files when it is called for those splits. One call per split is needed. `rebuild/build_vcc_caches.py`, written for the rebuild check, makes the three calls.
5. Four files of section 2 (under `IGA_MAIN/results/phase10/` and `IGA_MAIN/results/phase11b_classical_transfer/`) are saved predictions of models fitted on the Virtual Cell Challenge data: a constant, a linear model on magnitude summaries, ridge, gradient boosting and random forest models, and two methods that the article does not report. The code that wrote them is not included. Only part (c) of `bioinf2027/scripts/make_d1b.py` reads them, and no table or figure built by this repository uses the output of that part.
6. `phase2_embed_and_xcell.py` sets `CUDA_VISIBLE_DEVICES` to 1 in the code, and `phase3_runner.py` takes `--gpu` with default 1. Embeddings computed on a GPU do not repeat bit for bit (differences of about 4e-7 were measured), so rebuilt embedding files do not match the recorded checksums exactly.
7. The embedding scripts ran with the package versions of `bioinf2027/requirements_d2.txt` (anndata 0.11.4, torch 2.3.1, transformers 4.49.0). The other scripts also need anndata, except `phase2_reagg_and_oracle.py`, which needs numpy and pandas only.

## Rebuild check

The six steps that need no GPU were run again from the raw downloads, on a new data root that held only links to the ten raw files: on 2026-11-21 with the files as they were before the names were replaced, and on 2026-11-23 with the files of this copy. Both runs gave the same comparison with the stored files, and their rebuilt files were identical except one path field of the subsample sizes:

| Files | Result |
|---|---|
| gene names, gene selection, control baselines, batch order (four files) | identical (SHA-256) |
| the RPE1 and K562-essential eligibility lists and the K562-essential coverage file | identical (SHA-256) |
| the two score files, `rpe1_endpoint_by_construct.csv` and `endpoint_by_construct.csv` | 17 of 18 columns identical; the column `y_pooled` differs in 716 of 1,932 and in 774 of 1,903 values, by at most 2.66e-15 and 1.78e-15 |
| the RPE1 coverage file | the entry `rpe1` is identical; the rebuilt file has no entry `gwps` |
| the subsample sizes | identical except the recorded numpy version and one absolute path |
| the 144 per-batch files | every array identical; three fields of the zip headers differ |

Both reruns used Python 3.10.20, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3 and anndata 0.11.4. The original run of the score steps had used Python 3.13.12, numpy 2.4.4, scipy 1.17.1 and anndata 0.12.11. Each rerun took under five minutes and about 8 GB of memory. The record of the second rerun is `bioinf2027/diag/release_rebuild_check_renamed.json`. That record cites the record of the first rerun, which is not distributed, and the commit id of a copy made before this release. The first-version code of that copy is byte for byte the code of this folder.

To repeat it, make a data root that holds these ten files under `IGA_MAIN/data/`: `vcc/train/adata_Training.h5ad`, `vcc/train/pert_counts_Training.csv`, `vcc/validation/adata_Validation.h5ad`, `vcc/validation/pert_counts_Validation.csv`, `vcc/test/adata_Test.h5ad`, `vcc/test/pert_counts_Test.csv`, `replogle/raw/rpe1_normalized_bulk_01.h5ad`, `replogle/raw/K562_essential_normalized_bulk_01.h5ad`, `replogle/raw_singlecell/rpe1_raw_singlecell_01.h5ad` and `replogle/raw_singlecell/K562_essential_raw_singlecell_01.h5ad`. Create the two empty output folders, which the scripts do not create. Then run, from this folder:

```
export IGA_STORAGE_ROOT=/path/to/data_root
mkdir -p "$IGA_STORAGE_ROOT/IGA_MAIN/results_v4/harmonized_endpoint_v1"
mkdir -p "$IGA_STORAGE_ROOT/IGA_MAIN/results_v4/harmonized_endpoint_k562_essential_v1"

python -B rebuild/build_vcc_caches.py .
(cd scripts/v4/harmonized_endpoint_v1 && python -B step1_subsample_sizes.py && python -B step3_eligibility.py rpe1 && python -B run_replogle_endpoint.py rpe1)
(cd scripts/v4/harmonized_endpoint_k562_essential_v1 && python -B step_eligibility.py && python -B run_endpoint.py)
```

The first step writes about 2.9 GB under the data root.

## What the repository does about these files

The code under `bioinf2027/` does not trust the stored files blindly. It rebuilds the scores from the raw single-cell files and compares them with the stored ones, and it embeds cells again in the target-token analysis and compares unchanged inputs with the stored embeddings. See `docs/DATA.md`, section 3.
