# Analysis steps

Each analysis has a short code. The same codes are used in the file names, in the result folders and in the notation table of the supplement. `docs/COMMANDS.md` lists the exact commands of every analysis as they ran.

## Steps

| Code | What it does | Scripts in `bioinf2027/scripts/` | Saved results |
|---|---|---|---|
| D3 | cell counts and the groups kept at each minimum cell count | `make_d3_cell_counts.py`, `make_d3_vcc_retention.py` | `bioinf2027/diag/D3_*` |
| E1a | builds the all-cells score and the fixed-count scores, checks them against the stored scores, and computes the read-outs (association with cell number, agreement between cell halves) | `make_e1a_*.py` | `bioinf2027/endpoints/` |
| D1, D1b | what the target-gene score tracks: the drop in the target's expression and its expression in control cells | `make_d1_*.py`, `make_d1b*.py` | `bioinf2027/diag/D1*`, `D1b*` |
| E1b | models trained on the Virtual Cell Challenge with Geneformer features, expression features and their controls, on both scores; their values on the two external screens | `make_e1b_*.py`, `stage_*.py` | `bioinf2027/results/E1/` |
| D2 | removal of the target gene's token from the Geneformer input | `make_d2_*.py` | `bioinf2027/results/D2/` |
| E2 | the same comparison for scGPT, Stack and principal components; features from count-matched control cells; follow-up analyses on the external screens | `make_e2_*.py` | `bioinf2027/results/E2/` |
| E4 | features rebuilt from the cells of single draws | `make_e4_*.py` | `bioinf2027/results/E4/` |
| E6 | predictions between the two external screens with gene-disjoint folds | `make_e6_*.py`, `make_e2_fits_e6.py` | `bioinf2027/results/E6/` |
| E5 | prediction from gene identity: the score for changes in other genes, gene vectors and their controls, the gene-family mean and the DepMap fitness score | `make_e5_*.py` | `bioinf2027/results/E5/` |
| S1 supplement | the all-cells score and cell number on all three screens; subsampling within a group | `make_s1_supplement.py` | `bioinf2027/results/S1_supplement/` |
| closing | five small analyses from saved files | `make_missing_analyses_closing.py` | `bioinf2027/results/missing_analyses_closing/` |
| tables and figures | every table and figure from the saved results | `figures/table*.py`, `figures/fig*.py` | `bioinf2027/tables/`, `bioinf2027/figures/` |
| C1, D0 | comparisons with the first version of this work | `make_c1_checks.py`, `make_d0_checks.py` | need saved outputs of the first version, which are not distributed |

## Running a step

Run from the repository root. The launcher writes the log under the data root and appends a record (command, environment, start and end time) to `bioinf2027/runs/<code>.json`:

```
python bioinf2027/scripts/run_item.py --item D1b --label <label> bioinf2027/scripts/make_d1b.py
```

A script can also be run directly with `python bioinf2027/scripts/<script>.py`.

Several helper scripts only manage resources on a shared machine and change no result: `fit_slots.py`, `fit_tiers.py`, `memory_governor.py`, `memory_watch.py`, `memory_peaks.py`.

## Environments

| File | Used for | Interpreter |
|---|---|---|
| `bioinf2027/requirements.txt` | every step except the three embedding steps; tables and figures | Python 3.13.12 |
| `bioinf2027/requirements_d2.txt` | Geneformer embeddings (D2) | Python 3.10.20, PyTorch 2.3.1 |
| `bioinf2027/requirements_scgpt.txt` | scGPT embeddings (E2, E5); a layer on the D2 environment | Python 3.10.20 |
| `bioinf2027/requirements_stack.txt` | Stack embeddings (E2); a layer on the D2 environment | Python 3.10.20 |

The header of each file states how the environment was built.

## Seeds and resampling

Every random step takes its seed from the code. Interval estimates use 10,000 bootstrap draws of target genes (`bioinf2027/src/bioinf2027/config.py`, `stats.py`). Fixed-count scores are medians of 25 draws with the seeds recorded in `bioinf2027/endpoints/*/protocol.json`.

## Time and hardware

The full analysis ran on one Linux server over about ten days, most of it for model fits on CPU cores. The embedding steps used one GPU with 24 GB of memory. Level 1 (tables and figures from the saved results) takes a few minutes on a laptop.
