# crispri-cell-number-scores

Code, tests and saved summary statistics for the article

> Shoeibi M, Karwowski W, Yousefi N. Cell number and effect scores in the evaluation of CRISPRi perturbation models. Manuscript under review.

The article asks what a model comparison means when the score being predicted depends on how many cells were measured. It uses the Virtual Cell Challenge data (H1 cells) and two CRISPR interference screens (RPE1 and K562 cells).

## What this repository reproduces

There are three levels. Read this section before you plan a rerun.

| Level | From | To | Needs | Status |
|---|---|---|---|---|
| 1 | saved summary statistics (in this repository) | the tables and the figure data of the article | this repository only, a few minutes | every rebuilt CSV and LaTeX file is identical to the stored one (see "Check" below) |
| 2 | single-cell data, model weights and the intermediate files and folders listed in `docs/INPUT_FILES.md`, section 2 | the saved summary statistics | the data root described in `docs/DATA.md`; as run, about 65 GPU hours on one GPU with 24 GB and about 260 hours of wall-clock time on one Linux server | all code is here; commands in `docs/COMMANDS.md` |
| 3 | raw data | the intermediate files of the first version of this work (gene selection, the all-cells score file, Geneformer embeddings, expression-shift arrays, external eligibility lists) | code of the first version | partly here: `first_version/` holds the code that built 23 of these 34 files, as it ran apart from replaced names. The 11 files that need no GPU were rebuilt with it from the raw downloads and compared: 7 are identical; the two score files are equal except one column, which differs by at most 2.7e-15; the subsample sizes differ in two metadata fields; the RPE1 coverage file has an identical `rpe1` entry and lacks a second entry that needs a screen not downloaded here. The GPU steps were not rerun, and the other files have no code here; see `first_version/README.md` and `docs/DATA.md`, section 3 |

Raw single-cell data, model weights, embeddings and files with one row per group, construct or gene are not distributed here. `docs/DATA.md` says where each input comes from and gives its checksum.

## Quick start (level 1)

Python 3.13 is required. The analysis ran with Python 3.13.12.

```
python -m venv .venv
.venv/bin/pip install -r bioinf2027/requirements.txt

# the data root marker; an existing empty folder is enough for level 1
mkdir -p /tmp/data_root
echo /tmp/data_root > storage_root.txt

# only if you downloaded an archive instead of cloning with git:
# the table and figure scripts check their inputs against the git index
git init -q && git add -A
git -c user.name=local -c user.email=local@example.org commit -q -m "archive contents"

.venv/bin/python -m pytest
make tables figures PYTHON=.venv/bin/python
```

### Check

After `make tables figures`, every CSV, LaTeX and SVG output must be unchanged. The stored checksums are in `docs/OUTPUT_CHECKSUMS.sha256`:

```
make check PYTHON=.venv/bin/python
```

Two kinds of files do change and this is expected. The PDF files are redrawn and their bytes depend on the machine. The JSON record beside each output stores the Python version and the checksum of the PDF, so it changes with them. The numbers are in the CSV files.

## Layout

```
bioinf2027/
  src/bioinf2027/     library: scores, features, models, resampling, statistics, records
  scripts/            one script per analysis step; scripts/figures/ builds tables and figures
  tests/              unit tests (pytest)
  results/            saved summary statistics of every analysis, with JSON records
  endpoints/          score definitions, checks and read-outs
  diag/               data checks and cell counts
  tables/             tables of the article and supplement (CSV, LaTeX, record)
  figures/            figure data (CSV), figures (PDF) and records
  runs/               one record per run: command, environment, start and end time
  requirements*.txt   pinned environments
docs/
  DATA.md             data sets, downloads, data-root layout, what is not distributed
  INPUT_FILES.md      generated list of input files with sizes and SHA-256 checksums
  PIPELINE.md         analysis steps, their scripts and their place in the article
  COMMANDS.md         generated list of the commands of every analysis
  RECORDS.md          names used in the files and the names used in the article
  OUTPUT_CHECKSUMS.sha256   generated checksums of the tables and figure data
tools/
  check_outputs.py    compares rebuilt tables and figure data with the stored checksums
first_version/        code of the first version of this work, with its own README
```

The folder name `bioinf2027` and the variable `IGA_STORAGE_ROOT` are the names under which the analysis ran. Names of the authors' earlier working folders were replaced by neutral names in this copy: the data-root folders `IGA_MAIN` and `IGA_GENEFORMER_EXT`, the code folder `earlier_revision`, the Python constant `IGA_MAIN` of `first_version/iga_paths.py`, the names `PROJECT_2026-08-26`, `geneformer_env` and `C1_version_comparison`, and one word in four docstrings, now "revision". Where this replacement changed a file, the checksum and the size of that file recorded elsewhere in the copy were updated to match. Every other checksum, and every number, is the one recorded when the analyses ran. No table or figure output (CSV, LaTeX, SVG or PDF) changed. The JSON records beside some tables and one figure changed only in the checksums and sizes they record.

## Known limits

- `fig1_study_design.py`, `tableS_h_encoder_inputs.py` and `tableS_i_analysis_status.py` check quoted lines of the authors' dated analysis notes. Those notes are not distributed, so these three scripts do not run from this copy. Their stored outputs are included.
- `fig2_endpoint_cell_number.py --version supplement` plots one point per group and needs the score files under the data root (level 2).
- `make_c1_checks.py` and `make_d0_checks.py` compare with saved outputs of the first version of this work, which are not distributed.
- Two tests read the VCC training file. They are skipped when the file is absent.
- 22 tests compare this code with modules of the first version of this work. They look for those modules under `earlier_revision/` at the top of the repository, where this copy has none, so the tests are skipped. One of the modules, `x_cell_accumulator.py`, is under `first_version/earlier_revision/pipeline/`.
- Records contain fields such as `plan`, `gate` and review labels. They point to the authors' internal notes and logs (see `docs/RECORDS.md`).

## Licence and citation

The code is released under the MIT licence (`LICENSE`). Please cite the article and this repository (`CITATION.cff`).

The saved statistics derive from data released by their owners: the Arc Institute Virtual Cell Challenge data, the Replogle et al. (2022) Perturb-seq data (Figshare+, CC BY 4.0), DepMap 24Q4 Public (Figshare+, CC BY 4.0) and the GenePT gene vectors (Zenodo, CC BY 4.0). `docs/DATA.md` gives the references.

## Contact

Mehrdad Shoeibi, mehrdad.shoeibi@ucf.edu
