"""Table 1: datasets and cell counts (figure and table plan, Table 1; FS0 Section 1). Printed from saved files only:
bioinf2027/plans/E1_input_records.json, bioinf2027/diag/D3_vcc_retention.csv and bioinf2027/diag/D3_cell_counts.csv.
Usage, from the repository root: python bioinf2027/scripts/figures/table1_datasets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

REPO = tablelib.REPO
RECORDS = REPO / "bioinf2027" / "plans" / "E1_input_records.json"
RETENTION = REPO / "bioinf2027" / "diag" / "D3_vcc_retention.csv"
COUNTS = REPO / "bioinf2027" / "diag" / "D3_cell_counts.csv"


def n(x) -> str:
    return "{:,}".format(int(round(float(x))))


def q(median, q25, q75) -> str:
    fmt = lambda v: ("%.1f" % v).rstrip("0").rstrip(".")  # noqa: E731
    return "%s [%s, %s]" % (fmt(median), fmt(q25), fmt(q75))


def main() -> None:
    files = json.loads(RECORDS.read_text())["files"]
    retention = pd.read_csv(RETENTION)
    counts = pd.read_csv(COUNTS)
    rows = []
    for split, key in (("train", "VCC train"), ("val", "VCC validation"), ("test", "VCC test")):
        f = files[key]
        v = f["vcc"]
        r10 = retention[(retention.split == split) & (retention.perturbed_cell_floor == 10)].iloc[0]
        r27 = retention[(retention.split == split) & (retention.perturbed_cell_floor == 27)].iloc[0]
        c = counts[(counts.screen == "VCC") & (counts.split == split)].iloc[0]
        rows.append({"Dataset": {"train": "VCC training", "val": "VCC validation", "test": "VCC test"}[split],
                     "Cells": n(f["matrix"]["shape"][0]), "Genes": n(f["matrix"]["shape"][1]), "Control cells": n(v["control_cells"]),
                     "Perturbed cells": n(v["perturbed_cells"]), "Batches": n(v["batches"]), "Target genes": n(v["target_genes"]),
                     "Units": "%s (target gene, batch)" % n(v["groups"]),
                     "Units with at least 10 cells": "%s (%s genes)" % (n(r10.rows_retained), n(r10.target_genes_retained)),
                     "Units with at least 27 cells": "%s (%s genes)" % (n(r27.rows_retained), n(r27.target_genes_retained)),
                     "Cells per unit, at least 27": q(r27.cells_per_retained_row_median, r27.cells_per_retained_row_q25, r27.cells_per_retained_row_q75),
                     "Control cells per batch": q(c.control_per_batch_median, c.control_per_batch_min, c.control_per_batch_max).replace("[", "(").replace("]", ")")})
    for screen in ("RPE1", "K562-essential"):
        f = files[screen]
        c = counts[counts.screen == screen].iloc[0]
        rows.append({"Dataset": "Replogle " + screen, "Cells": n(f["matrix"]["shape"][0]), "Genes": n(f["matrix"]["shape"][1]),
                     "Control cells": n(c.control_pool_total), "Perturbed cells": "", "Batches": ("at least %s GEM groups" if str(c.batch_count_source).startswith("lower bound") else "%s GEM groups") % n(c.n_batches),
                     "Target genes": n(c.distinct_target_genes), "Units": "%s constructs" % n(c.denom2_post_join),
                     "Units with at least 10 cells": "", "Units with at least 27 cells": "%s (%s genes)" % (n(c.n_eligible), n(c.distinct_target_genes)),
                     "Cells per unit, at least 27": q(c.perturbed_median, c.perturbed_q25, c.perturbed_q75), "Control cells per batch": "pooled"})
    frame = pd.DataFrame(rows)
    tablelib.save("table1_datasets", frame, [RECORDS, RETENTION, COUNTS], Path(__file__).resolve(),
                  extra={"notes": {"Control cells per batch": "median (minimum, maximum) for VCC; the Replogle labels use pooled controls",
                                   "Cells per unit, at least 27": "median [first quartile, third quartile] of perturbed cells per retained unit",
                                   "Units": "VCC: (target gene, batch) groups with any perturbed cells; Replogle: constructs after the join to the gene matrix",
                                   "Target genes": "VCC: target_gene values recorded per file; Replogle: distinct target genes of the eligible constructs"}},
                  align="l" + "r" * 11)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: table1_datasets.py")
    main()
