# Commands of every analysis

This file is generated from the run records in `bioinf2027/runs/`. For each analysis it lists the
distinct commands that finished without error, in the order of their first run, with the number of
runs and the date of the last one. All commands run from the repository root.

## C1

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_c1_checks.py` | 2 | 2026-11-09 |

## D0

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_d0_checks.py` | 8 | 2026-11-09 |

## D1

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_d1_external.py` | 4 | 2026-11-09 |
| `python bioinf2027/scripts/make_d1_vcc_own_delta.py` | 4 | 2026-11-09 |
| `python bioinf2027/scripts/check_regeneration.py --item D1 --against a2bc766 --rename n=n_rows --rename n_groups=n_target_genes bioinf2027/diag/D1_label_vs_own_delta_vcc.csv bioinf2027/diag/D1_label_vs_own_delta_external.csv bioinf2027/diag/D1_target_in_features_vcc.csv bioinf2027/diag/D1_target_in_features_external.csv bioinf2027/diag/D1_target_in_features_external_per_construct.csv` | 1 | 2026-11-09 |
| `python bioinf2027/scripts/check_regeneration.py --item D1 --against 5bc8088 bioinf2027/diag/D1_label_vs_own_delta_vcc.csv bioinf2027/diag/D1_label_vs_own_delta_external.csv bioinf2027/diag/D1_target_in_features_vcc.csv bioinf2027/diag/D1_target_in_features_external.csv bioinf2027/diag/D1_target_in_features_external_per_construct.csv` | 1 | 2026-11-09 |
| `python bioinf2027/scripts/check_regeneration.py --item D1 --against 071b26e --tolerance 1e-12 --rename argmax_fraction=argmax_fraction_of_rows_with_target_feature --rename top2_fraction=top2_fraction_of_rows_with_target_feature --rename top5_fraction=top5_fraction_of_rows_with_target_feature --rename top10_fraction=top10_fraction_of_rows_with_target_feature --rename mapped_argmax_fraction=mapped_argmax_fraction_of_constructs_with_target_feature --rename mapped_top2_fraction=mapped_top2_fraction_of_constructs_with_target_feature --rename mapped_top5_fraction=mapped_top5_fraction_of_constructs_with_target_feature --rename mapped_top10_fraction=mapped_top10_fraction_of_constructs_with_target_feature bioinf2027/diag/D1_label_vs_own_delta_vcc.csv bioinf2027/diag/D1_label_vs_own_delta_external.csv bioinf2027/diag/D1_target_in_features_vcc.csv bioinf2027/diag/D1_target_in_features_external.csv bioinf2027/diag/D1_target_in_features_external_per_construct.csv` | 1 | 2026-11-09 |

## D1b

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_d1b.py` | 3 | 2026-11-10 |
| `python bioinf2027/scripts/make_d1b_rpe1_control_mean.py` | 1 | 2026-11-09 |

## D2

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_d2_input_records.py` | 2 | 2026-11-10 |
| `python bioinf2027/scripts/make_d2_units.py` | 2 | 2026-11-10 |
| `python bioinf2027/scripts/make_d2_embeddings.py --gpu 1 --identity` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_d2_embeddings.py --gpu 1 --identity --identity-name identity_single_precision_totals` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_d2_embeddings.py --gpu 0 --part vcc_delta --part rpe1_perturbed --part k562_perturbed --hours 22.7` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_d2_embeddings.py --gpu 0 --part k562_perturbed --part rpe1_delta --part k562_delta --hours 22.7` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_aggregate.py` | 2 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_fits.py --workers 3 --smoke` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_fits.py --workers 3` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_stats.py --smoke` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_stats.py` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_supplement.py --smoke` | 2 | 2026-11-13 |
| `python bioinf2027/scripts/make_d2_supplement.py` | 2 | 2026-11-13 |

## D3

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_d3_cell_counts.py` | 2 | 2026-11-09 |
| `python bioinf2027/scripts/make_d3_vcc_retention.py` | 1 | 2026-11-09 |
| `python bioinf2027/scripts/check_regeneration.py --item D3 --against 5bc8088 bioinf2027/diag/D3_cell_counts.csv` | 1 | 2026-11-09 |

## E1

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_e1_input_records.py` | 1 | 2026-11-09 |
| `python bioinf2027/scripts/make_e1a_extract.py` | 1 | 2026-11-09 |
| `python bioinf2027/scripts/make_e1a_r0.py` | 1 | 2026-11-09 |
| `python bioinf2027/scripts/make_e1a_label_extract.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1a_labels.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1a_readouts.py --smoke` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1a_readouts.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1a_readouts_supplement.py --smoke` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1a_readouts_supplement.py` | 2 | 2026-11-10 |
| `python bioinf2027/scripts/stage_external_control_means.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1b_input_records.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1b_checks.py --smoke` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1b_checks.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/diagnose_e1b_harness_controls.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/diagnose_e1b_harness_order.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1b_fits.py --smoke` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1b_stats.py --smoke` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e1b_fits.py` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e1b_stats.py` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/stage_e1b_check_inputs.py` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e1b_posthoc.py --smoke` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e1b_posthoc.py` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e1b_read_status.py` | 2 | 2026-11-13 |
| `python bioinf2027/scripts/make_e1b_attribution.py --fits` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e1b_attribution.py --stats` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e1b_attribution_amendment.py --fits --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e1b_attribution_amendment.py --stats --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e1b_attribution_amendment.py --fits` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e1b_attribution_amendment.py --stats` | 1 | 2026-11-14 |

## E2

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_e2_pretraining_check.py` | 2 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_downloads.py stack_large` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_scgpt.py --probe A` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_scgpt.py --probe B` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_scgpt.py --probe R` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_scgpt.py --compare` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_stack.py --probe A` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_stack.py --probe B` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_stack.py --compare` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_stack.py --stage VCC RPE1 K562-essential` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_stack.py --units VCC RPE1 K562-essential` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_features.py pca geneformer_null` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_features.py stack` | 2 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_fits.py stack pca50 pca768 --geneformer-null --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_embed_scgpt.py --part K562-essential_control K562-essential_perturbed RPE1_control RPE1_perturbed vcc_control vcc_perturbed_test vcc_perturbed_train vcc_perturbed_val` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_features.py scgpt` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_features.py pca50_null` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_order_sensitivity.py` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_coverage.py` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e2_fits.py stack pca50 pca768 --geneformer-null --smoke --workers 2 --workers-file bioinf2027/control/e2-fits-smoke.workers` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/check_e2_workers_change.py` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/check_e2_slots_cache.py E2_fits_smoke_before_slots E2_fits_smoke_ref_7f36b1d ref_7f36b1d_subset --subset` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e2_fits.py stack pca50 pca768 --geneformer-null --workers 3 --workers-file bioinf2027/control/e2-fits.workers` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e2_fits.py stack pca50 pca768 --geneformer-null --smoke --slots --workers 4 --workers-file bioinf2027/control/e2-smoke-slots.workers` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/check_e2_slots_cache.py E2_fits_smoke_before_slots E2_fits_smoke slots_full` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e2_fits.py scgpt --smoke --slots --workers 4 --workers-file bioinf2027/control/e2-smoke-scgpt.workers` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e2_fits.py stack pca50 pca768 --geneformer-null --smoke --slots --workers 4 --workers-file bioinf2027/control/e2-smoke-e0rec.workers` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/check_e2_slots_cache.py E2_fits_smoke_slots_97cee0d E2_fits_smoke e0_record --reference-files` | 2 | 2026-11-15 |
| `python bioinf2027/scripts/make_e2_fits.py scgpt --slots --workers 3 --workers-file bioinf2027/control/e2-fits-scgpt.workers` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_fits_e6.py scgpt stack pca50 pca768 --smoke` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_fits_e6.py scgpt stack pca50 pca768 --smoke --slots --workers 4 --workers-file bioinf2027/control/e2-e6-smoke-slots.workers` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/check_e2_slots_cache.py E2_e6_smoke_ref_c05855c E2_e6_smoke_slots_1dae755 e6_slots` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_fits_e6.py scgpt stack pca50 pca768 --slots --workers 3 --workers-file bioinf2027/control/e2-fits-e6.workers` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_stats.py --smoke` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_stats.py --workers 6` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_scgpt_input_share.py --workers 4` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/check_e2_input_digests.py` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_supplement.py --smoke` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_null_external.py --features` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_supplement.py` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_scgpt_target_input.py --workers 3` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_null_external.py --fits --smoke --workers 3 --workers-file bioinf2027/control/e2-null-smoke.workers` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_fits.py --check-e0-only --fresh-e0 --verify-inputs` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_null_external.py --fits --workers 4 --workers-file bioinf2027/control/e2-null-fits.workers` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_null_external.py --stats --workers 4` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e2_null_external_followup.py --saved --workers 4` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e2_null_external.py --fits --smoke --label existing --workers 3 --workers-file bioinf2027/control/e2-null-existing-smoke.workers` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e2_null_external.py --fits --label existing --workers 4 --workers-file bioinf2027/control/e2-null-existing-fits.workers` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e2_null_external.py --stats --label existing --workers 4` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e2_null_external_followup.py --saved --out-scratch E2_null_followup_default_check --workers 4` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e2_null_external_followup.py --saved --label existing --workers 4` | 1 | 2026-11-17 |

## E4

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_e4_extract.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/diagnose_e4_external_z.py` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e4_features.py` | 2 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_features.py --thinned` | 2 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_fits.py --smoke` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_fits.py --smoke --tier 1` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_fits.py --smoke --tier 2` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_fits.py --smoke --tier 3` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_stats.py --smoke --pass family` | 3 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_stats.py --smoke --pass full` | 2 | 2026-11-11 |
| `python bioinf2027/scripts/make_e4_fits.py --tier 1 --workers 3` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e4_fits.py --tier 2 --workers 3` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_e4_stats.py --pass family --smoke` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_e4_stats.py --pass family` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_e4_fits.py --tier 3 --smoke` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e4_fits.py --tier 3 --smoke --slots --smallest-first --workers-file bioinf2027/control/e4-smoke-slots.workers` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/check_e2_slots_cache.py E4_fits_smoke_ref_21ccc4c E4_fits_smoke_slots_a9d1e0d e4_slots` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/check_e2_slots_cache.py E4_fits_smoke_20261111 E4_fits_smoke_slots_a9d1e0d e4_slots_vs_20261111 --subset` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e4_fits.py --tier 3 --slots --smallest-first --workers-file bioinf2027/control/e4-tier3.workers` | 1 | 2026-11-18 |
| `python bioinf2027/scripts/make_e4_stats.py --pass full` | 1 | 2026-11-18 |

## E5

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_e5_endpoint_b.py --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_identity_vectors.py` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_endpoint_b.py` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_endpoint_b_vcc.py --smoke` | 2 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_identity_vectors_downloaded.py` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_fits.py --prefit --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_fits.py --prefit` | 2 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_fits.py --positive-control-revised --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_fits.py --positive-control-revised` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_fits.py --family --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_endpoint_b_vcc.py --workers 2` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e5_secondary.py --smoke --workers 2 --part vcc --part size --part prefit50 --part main` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e5_secondary.py --workers 2 --part vcc --part size --part prefit50` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e5_fits.py --family` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e5_secondary.py --smoke --workers 4 --part main` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/check_e5_saving_smoke.py E5_secondary_smoke_saving_20f5f53 E5_secondary_smoke_ref_7062cdf saving_smoke` | 1 | 2026-11-15 |
| `python bioinf2027/scripts/make_e5_secondary.py --smoke --slots --workers 4 --workers-file bioinf2027/control/e5-smoke-slots.workers --part main` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/check_e5_saving_smoke.py E5_secondary_smoke E5_secondary_smoke_saving_20f5f53 slots_smoke` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/check_e5_saving_smoke.py E5_secondary_smoke E5_secondary_smoke_ref_7062cdf slots_smoke_vs_7062cdf` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e5_secondary.py --slots --workers 4 --workers-file bioinf2027/control/e5-secondary-main.workers --part main` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e5_secondary.py --smoke --workers 4 --part forest` | 1 | 2026-11-16 |
| `python bioinf2027/scripts/make_e5_secondary.py --workers 8 --part forest` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/check_e5_input_digests.py` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e5_depmap_download.py` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e5_supplement.py --smoke --workers 3 --workers-file bioinf2027/control/e5-supplement-smoke.workers --part depth --part lists --part breakdown --part fits --part pool --part depmap` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e5_supplement.py --workers 4 --workers-file bioinf2027/control/e5-supplement.workers --part depth --part lists --part breakdown --part fits --part pool --part depmap` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e5_hgnc_download.py` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e5_supplement_2.py --smoke --workers 3 --workers-file bioinf2027/control/e5-supplement-2-smoke.workers --part hgnc --part fits --part route --part b0` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_e5_supplement_2.py --workers 4 --workers-file bioinf2027/control/e5-supplement-2.workers --part hgnc --part fits --part route --part b0` | 1 | 2026-11-17 |

## E6

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_e6_fits.py --smoke` | 1 | 2026-11-10 |
| `python bioinf2027/scripts/make_e6_stats.py --smoke` | 2 | 2026-11-10 |
| `python bioinf2027/scripts/make_e6_fits.py --smoke --tier 1` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e6_fits.py --smoke --tier 2` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e6_fits.py --smoke --tier 3` | 1 | 2026-11-11 |
| `python bioinf2027/scripts/make_e6_stats.py --smoke --pass family` | 2 | 2026-11-11 |
| `python bioinf2027/scripts/make_e6_stats.py --smoke --pass full` | 2 | 2026-11-11 |
| `python bioinf2027/scripts/make_e6_fits.py --check` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e6_fits.py --tier 1 --workers 9` | 1 | 2026-11-12 |
| `python bioinf2027/scripts/make_e6_fits.py --tier 2 --workers 9` | 2 | 2026-11-13 |
| `python bioinf2027/scripts/make_e6_fits.py --tier 2 --smoke` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_e6_stats.py --pass family --smoke` | 2 | 2026-11-13 |
| `python bioinf2027/scripts/make_e6_stats.py --pass family` | 1 | 2026-11-13 |
| `python bioinf2027/scripts/make_e6_input_records.py` | 2 | 2026-11-14 |
| `python bioinf2027/scripts/make_e6_fits.py --tier 3 --workers 8` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e6_supplement.py` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e6_stats.py --pass full` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e6_supplement.py --smoke` | 1 | 2026-11-14 |
| `python bioinf2027/scripts/make_e6_supplement_g3.py` | 1 | 2026-11-14 |

## S1

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_s1_supplement.py --smoke --workers 2 --part vcc --part external --part control` | 1 | 2026-11-17 |
| `python bioinf2027/scripts/make_s1_supplement.py --workers 4 --part vcc --part external --part control` | 1 | 2026-11-17 |

## closing

| Command | Runs | Last finished |
|---|---|---|
| `python bioinf2027/scripts/make_missing_analyses_closing.py` | 2 | 2026-11-17 |
