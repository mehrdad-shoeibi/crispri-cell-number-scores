# Input files and checksums

This file is generated from the JSON records of the repository. Every row is a file that a record
names with its SHA-256 checksum. Paths are relative to the data root (see `docs/DATA.md`).

## 1. Downloaded files

29 files: the single-cell data, the model weights and the other public downloads.

| Path under the data root | Bytes | SHA-256 |
|---|---|---|
| `IGA_MAIN/data/replogle/raw/K562_essential_normalized_bulk_01.h5ad` | 79,766,954 | `c1ca6456c9c9f1aa2b02c496eb64d1dc3e6a852edbd744d682b8d2c95fd36829` |
| `IGA_MAIN/data/replogle/raw/rpe1_normalized_bulk_01.h5ad` | 95,350,546 | `a3c5bfd0f15d63938bc80c9b8874b9cd761e3a23caf5ffe7966bae4e887ec89d` |
| `IGA_MAIN/data/replogle/raw_singlecell/K562_essential_raw_singlecell_01.h5ad` | 10,661,879,995 | `3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee` |
| `IGA_MAIN/data/replogle/raw_singlecell/rpe1_raw_singlecell_01.h5ad` | 8,700,873,216 | `9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1` |
| `IGA_MAIN/data/vcc/test/adata_Test.h5ad` | 11,950,739,168 | `ba5ce66729e4c8e50b6f147c7d73d3f9f28f76c91fa43ff33e3f6bc2ed831f03` |
| `IGA_MAIN/data/vcc/train/adata_Training.h5ad` | 15,482,497,461 | `a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b` |
| `IGA_MAIN/data/vcc/validation/adata_Validation.h5ad` | 6,928,967,541 | `376f0bab27d9f22e00dfd888668d3602006d035edc182721c378f4802c77bfe1` |
| `MADA/cache/foundation_models/geneformer-repo/Geneformer-V2-104M/config.json` | 590 | `467d4492f0dd53b4d60afffe20812db484ca1cf9fdbeb6a6e060e93564f70859` |
| `MADA/cache/foundation_models/geneformer-repo/Geneformer-V2-104M/model.safetensors` | 417,571,156 | `fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/emb_extractor.py` | 35,355 | `50d4eee1677b376f4d2c50887558450fdd39150c4ed902833e9929f25f17adb0` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/ensembl_mapping_dict_gc104M.pkl` | 3,957,652 | `0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/gene_median_dictionary_gc104M.pkl` | 1,512,661 | `a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/gene_name_id_dict_gc104M.pkl` | 1,660,882 | `fabfa0c2f49c598c59ae432a32c3499a5908c033756c663b5e0cddf58deea8e1` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/perturber_utils.py` | 32,453 | `7a6db4f0ddf3a0877c698145948ecff1806e35cd719f0e28b753407099d02ae5` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/token_dictionary_gc104M.pkl` | 425,590 | `67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f` |
| `MADA/cache/foundation_models/geneformer-repo/geneformer/tokenizer.py` | 34,686 | `689b71a916b75fa618fbb460a7fc460c3ab32d41e4f98064efb0ebb3ee921002` |
| `bioinf2027/downloads/depmap_24Q4/CRISPRGeneEffect.csv` | 428,678,699 | `3d8f3ec6dbf2db7ff834b79b508622ec0b226f3518003fe96ecf5a4fcf167e3b` |
| `bioinf2027/downloads/depmap_24Q4/Model.csv` | 645,696 | `b7a0c1385e6cef30132b56aff61f1261d11e3f490490b355c430d32ee0dbdcfa` |
| `bioinf2027/downloads/genept_v2/GenePT_emebdding_v2.zip` | 574,395,233 | `6193575dcbd7bbf214c8ca3eb518cb3d13272443f98ecd6c26402411a7ca745e` |
| `bioinf2027/downloads/genept_v2_extracted/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle` | 460,797,248 | `fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862` |
| `bioinf2027/downloads/hgnc/hgnc_complete_set.txt` | 17,007,798 | `6df77ac3cbaeb86a6dce5e17d851ebb27b6b3cdcd2471a211a3d8464f00c54c4` |
| `bioinf2027/downloads/scbasecount_2025-02-25/sample_metadata.parquet` | 532,899 | `6721132bca67652901a32087077a5ef552becb0053ec7faed67cc617a1e17d61` |
| `bioinf2027/downloads/scbasecount_2026-01-12/sample_metadata.parquet` | 3,205,158 | `e227fb5e1760b64bd54e04e76677891b935ba2c2cbec980f157ca64990719659` |
| `bioinf2027/downloads/scgpt_whole_human/args.json` | 1,300 | `c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60` |
| `bioinf2027/downloads/scgpt_whole_human/best_model.pt` | 205,385,258 | `6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922` |
| `bioinf2027/downloads/scgpt_whole_human/vocab.json` | 1,317,639 | `acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f` |
| `bioinf2027/downloads/stack_large/MODEL_LICENSE.md` | 9,863 | `53bb1eea6fc3e8e5506846b97b1fd2c8f828b6037d14b582511fbeeeec56c623` |
| `bioinf2027/downloads/stack_large/basecount_1000per_15000max.pkl` | 925,039 | `d8761dfda955b9897d3251798b72361ddd0171ef707119eaf65381bed2d85dcc` |
| `bioinf2027/downloads/stack_large/bc_large.ckpt` | 2,610,004,146 | `d388c75891fb26618e6e36054c9819b6a82d6f5bc608baac75994e33ddc8bb81` |

## 2. Intermediate files made by the first version of this work

34 files. The code of this repository reads them and does not build them (see `docs/DATA.md`, section 3).

| Path under the data root | Bytes | SHA-256 |
|---|---|---|
| `IGA_GENEFORMER_EXT/checkpoints/rpe1_xcell/xcell_state.pkl` | 406,185,049 | `7f4e231e050439fcd69815e6972bc54a432894d0ae2ccd2a048eca29d84de336` |
| `IGA_GENEFORMER_EXT/k562_essential/checkpoints/xcell_state.pkl` | 391,592,139 | `74bad0712da3fd6037a27014a9b1316010cfad913f1d6772aeef999904c32175` |
| `IGA_GENEFORMER_EXT/k562_essential/x_cell/x_cell_k562_1903_mapped.npy` | 15,224,128 | `326142005d94fadf682a5d71a2dfaea5ba40ba4a5f40e61b2918e6f74f374965` |
| `IGA_GENEFORMER_EXT/k562_essential/x_cell/x_cell_k562_native.npz` | 130,987,790 | `f8127fbc14cdefd340235da4815908a3b5a4a14b3c8f0456e8ac7d60c9e765f2` |
| `IGA_GENEFORMER_EXT/k562_essential/z/k562_construct_index.parquet` | 47,129 | `8d5f3e8aec2471e797dfa74c090496579a8351ba9feed35440df540c5821bbf7` |
| `IGA_GENEFORMER_EXT/k562_essential/z/z_k562_1903.npy` | 5,846,144 | `ce88160c8a525b3ea4608cfc8978ee0f71e24dde3d6fb0c5335f0bcf3f790a26` |
| `IGA_GENEFORMER_EXT/manifests/phase2_rpe1_manifest.json` | 5,329 | `f1c78a931653edfc9890a63760fc2f2abdffb51c74d6f338dd60be352a4f1aec` |
| `IGA_GENEFORMER_EXT/rpe1_topup/rpe1_construct_index.parquet` | 48,467 | `37f05c17c268f039bfcf011ccde1a70b0e9eb020b9c60298b03c0892daed43ec` |
| `IGA_GENEFORMER_EXT/rpe1_topup/z_rpe1_1932.npy` | 5,935,232 | `f115a777bd78f644219189b6589086978c2b57fd51b32d9ece39c5ba42fdd904` |
| `IGA_GENEFORMER_EXT/scratch/nt_oidx.npy` | 92,008 | `7daa49d62f35be91b8c8645b19310aff8fa3d7458185732625063b224d3ea696` |
| `IGA_GENEFORMER_EXT/x_cell/x_cell_rpe1_1932_mapped.npy` | 15,456,128 | `d14cd7e0cdddc04ea81ea6ddf149ee1bb01fc091ebdf6cf3489bc274d5bd33c8` |
| `IGA_GENEFORMER_EXT/x_cell/x_cell_rpe1_mapped.npz` | 16,229,294 | `ddc3cab022073333f3307f06f08ceab64a4f45b453f4a95ff721452dee06fe1e` |
| `IGA_GENEFORMER_EXT/x_cell/x_cell_rpe1_native.npz` | 135,997,838 | `ee766e65c18cace39b7a7d225b3e218d22330b564b765759c6f63371d3e3ec57` |
| `IGA_MAIN/cache/vcc/nt_baseline/batch_order.npy` | 853 | `d085c11090e1796677b94141607b2ed755c278fb86519b0fa7a9cf249ddc7090` |
| `IGA_MAIN/cache/vcc/nt_baseline/mean_log1p.npy` | 3,471,488 | `fd811b8598fd43acffad76651b072707a039b48bd1d6f5b966016858e6dfa21b` |
| `IGA_MAIN/cache/vcc/var_gene_names.npy` | 152,515 | `9f35391f26907834f50e881c7d3fb32351cc6a3c5449230436731182d7c91c85` |
| `IGA_MAIN/cache/vcc/variants/delta_hvg/selected_genes.npy` | 8,128 | `805d17402aa2804f843ca9f57459bb85d50c067df02a3d214ee5b5ee4229bb01` |
| `IGA_MAIN/results/phase10/transfer_run/replogle_predictions.csv` | 185,551 | `c496b9618389bc395d075f1bfe98b0eebe33bd10036bf38c6636637a764e71b6` |
| `IGA_MAIN/results/phase10/transfer_run_RPE1/replogle_predictions.csv` | 218,408 | `0b3f14b12e757cdeecb9c6bf070b15c24ef4f6bbe7e5025c751d9d4fdf1a98cf` |
| `IGA_MAIN/results/phase11b_classical_transfer/replogle_predictions.csv` | 884,968 | `2e7c64d702fef5513735ea6a3e98b911d2c3e65bde9268a4aeff9d864f5f4cb3` |
| `IGA_MAIN/results/phase11b_classical_transfer/replogle_predictions_RPE1.csv` | 1,044,675 | `6a22af0b8621090db37990b506fe2b7abcdf1fd8542a927fbbce112e6e10a0ed` |
| `IGA_MAIN/results_v4/harmonized_endpoint_k562_essential_v1/coverage_summary.json` | 902 | `93634aafe2405b5b56085628218b488c6d61c67f2fff49fd46cb23d1da89e05c` |
| `IGA_MAIN/results_v4/harmonized_endpoint_k562_essential_v1/eligibility.csv` | 190,992 | `b10de8bfbf95c825efeaa6f562bb0ed9981f55897242ce91c5c22f7b773f9158` |
| `IGA_MAIN/results_v4/harmonized_endpoint_k562_essential_v1/endpoint_by_construct.csv` | 433,586 | `051269eeeae93c75c84ca1fc050529a096ebd15e1990cb14bb24669c1b9672ff` |
| `IGA_MAIN/results_v4/harmonized_endpoint_v1/coverage_summary.json` | 1,721 | `dfd586a2b0fd0d425c023d72e8614b5e58e22fedd625dce1fda5d1aa4e50f8a8` |
| `IGA_MAIN/results_v4/harmonized_endpoint_v1/rpe1_eligibility.csv` | 228,919 | `71da80a16e6cd982fd280b616675106f770ad01f8aeeb5a0aa309fdf996ba071` |
| `IGA_MAIN/results_v4/harmonized_endpoint_v1/rpe1_endpoint_by_construct.csv` | 440,592 | `353c04d3ac8e3380a3addf3abc1953d30799423f7cd467006adc36800e26a126` |
| `IGA_MAIN/results_v4/harmonized_endpoint_v1/subsample_sizes.json` | 3,846 | `d6068e3b7106fb89d1c4f6b7d3d5bf0fa8e26ee2f0207e4660a08966ae7562dd` |
| `IGA_MAIN/results_v4/k562_essential_coverage_gate_v1/gemgroup_nt.csv` | 598 | `c6ef19cc6f64c4914d03b370f265e9359396cdb1c0f28b3c988d7df4faf8b00f` |
| `IGA_MAIN/results_v4/transfer_pretest_lock_v1/arm1_rpe1_construct_list.csv` | 92,747 | `e4d7534822a32564eaae0ed16f4e3dfa88616a00ecd2d7284140155b2985e3b2` |
| `MADA/cache/foundation_models/geneformer/full_feature_extraction_v2/cell_embedding_index.parquet` | 4,331,040 | `1b077612bf27b6489fb35399010237ed869ed1e9f2b8f11f57073662f8d83b85` |
| `MADA/cache/foundation_models/geneformer/full_feature_extraction_v2/geneformer_row_features_full.parquet` | 132,134,861 | `c531f392924076f2ca0ff360199ecbe3a323478933ce5bdc63de19bba9c81303` |
| `MADA/data/metadata/vcc/final_labels.parquet` | 989,164 | `37c9aafef6e721815436a6534eacda4a1db6e73e937e886c44588284a37a0fa0` |
| `MADA_data/processed/phasec/rpe1_features/full/master_cell_index_v1.parquet` | 6,155,772 | `e7c1d22cc8f8edc53fcaac1570152c8e1dd119f3dcde1fcaab5024f4bb54a459` |

9 folders of the first version are read as a whole. The records give the number of files and,
where it was taken, the checksum of the file listing. A folder and its sub-folders can both appear.

| Folder under the data root | Files | Bytes | SHA-256 of the file listing |
|---|---|---|---|
| `IGA_GENEFORMER_EXT/k562_essential/emb_units/` | 474 | 879,987,142 | `91443d5f7a01669fbb642f6dbb369e79a7088fd76b2a9ff7978f3a31499dba55` |
| `IGA_GENEFORMER_EXT/rpe1_topup/` | 4 | 32,042,514 | `ce6d4ff205a24a686a81fba979ac1182250c1362defc1af1061a5ec3a5f46016` |
| `IGA_MAIN/cache/vcc/variants/delta_hvg/shards/` | 144 | 94,609,758 | `74e6232f38df9dba498581435b97269c88c32f450f47dfbfd4d96e58ff48e3b8` |
| `IGA_MAIN/cache/vcc/variants/delta_hvg/shards/test/` | 48 | 31,942,613 | `a009ca286a47948f359b05aee7a83442ed07305ea3140ce9e4d029f7e35cb9ce` |
| `IGA_MAIN/cache/vcc/variants/delta_hvg/shards/train/` | 48 | 46,122,271 | `b4d5980372787ea9bbb1c6aa7d55b97ccb5be7d2c45627531d0cd7284c05b3b1` |
| `IGA_MAIN/cache/vcc/variants/delta_hvg/shards/val/` | 48 | 16,544,874 | `962814a1e7808556f58dcdc2776b1f771352bb23f27ceb67bc373c2de42f15d3` |
| `MADA/cache/foundation_models/geneformer/full_feature_extraction_v2/nt_embeddings/` | 144 | 119,652,944 | `a2cbf672c1aef76db2a4d7c19ec828f4375ccb562ee565552dfc8efbcebf165a` |
| `MADA/cache/foundation_models/geneformer/full_feature_extraction_v2/perturbed_embeddings/` | 432 | 1,182,242,976 | `261604fb1a88c5d46dd41d1d8da8b7675c9b93eab6946246f1a9589f483792f8` |
| `MADA_data/processed/phasec/rpe1_features/full/geneformer_chunks/` | 432 | 12,771,696,060 | `31f2a410706c596bf63d72497e38c543b2d041a7b5c2b28ba8cf97e60ca13efd` |

## 3. Intermediate files made by the code of this repository

20579 files have a recorded checksum (embeddings, features, score files and saved predictions).
They are too many to list here. The records under `bioinf2027/results/` name each one with its
checksum, for example the `predictions_manifest*.json` files. Number of files per folder of the data root:

| Folder under the data root | Files with a recorded checksum |
|---|---|
| `bioinf2027/D2/arms` | 1 |
| `bioinf2027/D2/embeddings` | 2 |
| `bioinf2027/D2/predictions` | 196 |
| `bioinf2027/D2/units` | 7 |
| `bioinf2027/E1b/attribution` | 108 |
| `bioinf2027/E1b/inputs` | 2 |
| `bioinf2027/E1b/predictions` | 3,773 |
| `bioinf2027/E2/embeddings` | 8 |
| `bioinf2027/E2/features` | 5 |
| `bioinf2027/E2/null_external` | 1 |
| `bioinf2027/E2/predictions` | 784 |
| `bioinf2027/E2/predictions_e6` | 390 |
| `bioinf2027/E2/predictions_null_external` | 24 |
| `bioinf2027/E2/predictions_null_external_existing` | 24 |
| `bioinf2027/E4/features` | 53 |
| `bioinf2027/E4/predictions` | 1,552 |
| `bioinf2027/E5/identity` | 3 |
| `bioinf2027/E5/secondary` | 2 |
| `bioinf2027/E5/supplement` | 12 |
| `bioinf2027/E5/supplement_2` | 36 |
| `bioinf2027/E6/predictions` | 13,508 |
| `bioinf2027/cache/E1a` | 6 |
| `bioinf2027/cache/E1a_labels` | 2 |
| `bioinf2027/cache/E4` | 1 |
| `bioinf2027/cache/e1b_check_inputs` | 10 |
| `bioinf2027/cache/external_control_means` | 2 |
| `bioinf2027/endpoints/G1` | 7 |
| `bioinf2027/endpoints/G1-DT-K562-essential` | 1 |
| `bioinf2027/endpoints/G1-DT-RPE1` | 1 |
| `bioinf2027/endpoints/G1-SU-K562-essential` | 2 |
| `bioinf2027/endpoints/G1-SU-RPE1` | 2 |
| `bioinf2027/endpoints/G2` | 7 |
| `bioinf2027/endpoints/G3` | 1 |
| `bioinf2027/endpoints/G4` | 1 |
| `bioinf2027/endpoints/G5` | 7 |
| `bioinf2027/endpoints/G5-DT-K562-essential` | 1 |
| `bioinf2027/endpoints/G5-DT-RPE1` | 1 |
| `bioinf2027/endpoints/G6` | 7 |
| `bioinf2027/endpoints/G6-DT-K562-essential` | 1 |
| `bioinf2027/endpoints/G6-DT-RPE1` | 1 |
| `bioinf2027/endpoints/supplement` | 12 |
| `bioinf2027/scratch/E2_e6_smoke_ref_c05855c` | 1 |
| `bioinf2027/scratch/E2_e6_smoke_slots_1dae755` | 1 |
| `bioinf2027/scratch/E2_fits_smoke` | 1 |
| `bioinf2027/scratch/E2_fits_smoke_before_slots` | 1 |
| `bioinf2027/scratch/E2_fits_smoke_slots_97cee0d` | 1 |
| `bioinf2027/scratch/E2_scgpt_draws` | 3 |
| `bioinf2027/scratch/E4_fits_smoke_20261111` | 1 |
| `bioinf2027/scratch/E4_fits_smoke_ref_21ccc4c` | 1 |
| `bioinf2027/scratch/E4_fits_smoke_slots_a9d1e0d` | 1 |
| `bioinf2027/scratch/E4_stats_smoke` | 2 |
| `bioinf2027/scratch/E6_stats_smoke` | 2 |

## 4. Files recorded with more than one checksum

These files were rebuilt during the work; each record states the checksum of the version it read.

- `bioinf2027/scratch/E2_fits_smoke/predictions_manifest_stack_pca50_pca768_gfnull.json`
