# Names used in the files and in the article

The files use the short names under which the analysis ran. The article uses plain words. This page maps one to the other and explains the records.

## Scores and settings

| In the files | In the article |
|---|---|
| existing label | all-cells score |
| rebuilt label, size-fixed label | fixed-count score |
| endpoint (a) | target-gene score |
| endpoint (b) | score for changes in other genes |
| G1 | 27 perturbed cells against 788 pooled controls |
| G2 | 10 perturbed cells against 788 pooled controls |
| G5 | 27 perturbed cells against 700 controls (same batch in the Virtual Cell Challenge, pooled in the external screens) |
| G6 | 10 perturbed cells against 700 controls |
| G3, G4 | the 27-cell and 10-cell scores with one row per target gene |
| row | group: one target gene in one batch (Virtual Cell Challenge) or one guide construct (external screens) |

## Inputs of the models

| In the files | In the article |
|---|---|
| z | Geneformer cell features |
| e, encoder, embedding | a cell representation (Geneformer, scGPT, Stack or principal components) |
| x | expression-shift profile over 2,000 genes |
| m | four summaries of the expression-shift profile |
| C1 | Gaussian features of the same width |
| C2 | random projection of the expression-shift profile |
| C3 | the representation with its rows shuffled |
| N0, size-matched null | features from count-matched control cells |
| B0 | logarithm of one plus the perturbed-cell count |
| B1 | the target's mean expression in control cells |
| B2, own shift | drop in the target's mean expression |
| B3, B4 | change in the fraction of cells that detect the target; change in its spread |
| TR | B0, B1 and B2 together |
| Bc | the target's control-cell profile in the gene-identity analysis |
| C, S (gene vectors) | Gaussian vectors; vectors reassigned to other genes |
| FN | gene-family mean (gene-name prefix) |
| F_mean, F_K562 | DepMap fitness score: mean over cell lines; K562 value |

## Statistics

| In the files | Meaning |
|---|---|
| T | external rank correlation of a representation minus that of Gaussian features |
| L | T after training on the fixed-count score minus T after training on the all-cells score |
| R | rank agreement of prediction and score after removing the five simple features from both |
| R5 | agreement of the residual score between halves of the cells |
| P-a, P-r, P-d | gain over Gaussian features on the all-cells score, on the fixed-count score, and their paired difference |

## JSON records

Beside most result files there is a JSON record. It lists the input files with their SHA-256 checksums, the code files with their checksums, the environment and the settings. Paths that start with `$IGA_STORAGE_ROOT/` are under the data root (`docs/DATA.md`). Other paths are relative to the repository root.

`bioinf2027/runs/<code>.json` holds one entry per run: the command, the environment, the start and end time, and the commit of the authors' working repository at that time. That working history is not part of this repository, so the commit identifiers are for the authors' own reference.

## References to notes that are not distributed

Docstrings, comments and records name files such as `bioinf2027/plans/<name>.md`, `bioinf2027/reports/<name>.md`, `bioinf2027/factsheets/<name>.md` and `bioinf2027/GATE.md`, and use labels such as "gate", "review" and "audit". These are the authors' dated analysis notes and their internal checking steps. They are not part of this repository. A reference of this kind tells you which decision or check a line of code follows; the code does not need the note to run, except for the three scripts listed in the README.
