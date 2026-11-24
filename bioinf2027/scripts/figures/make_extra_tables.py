"""Build four supplementary tables from saved statistics. No model is fitted.

Tables written to the output directory, each as CSV and LaTeX, with one JSON
record that lists the input files and their SHA-256 digests:

  vcc_predictor_sensitivity   VCC performance by predictor and minimum cell count
  vcc_absolute_r2             absolute test R2 per feature set, both scores
  vcc_crossed_intervals       gene, batch and crossed gene-by-batch intervals
  split_half_agreement        agreement of the fixed-count score between cell halves

Usage:
  python make_extra_tables.py --repo PATH_TO_REPOSITORY --out OUTPUT_DIRECTORY
"""
import argparse
import csv
import hashlib
import json
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

VCC = "bioinf2027/results/E2/stats/vcc.csv"
CROSSED = "bioinf2027/results/E2/supplement/crossed.csv"
HALVES = "bioinf2027/endpoints/readouts/supplement/5_6_R5_detail.csv"

INPUTS = [("geneformer", "Geneformer"), ("scgpt", "scGPT"), ("stack", "Stack"), ("pca50", "PCA-50")]
HEADS = [("forest", "Random forest"), ("boosting", "Gradient boosting"), ("ridge", "Ridge")]
CELLS = [("G6", "10"), ("G5", "27")]
SCORES = [("existing", "All cells"), ("rebuilt", "Fixed count")]
ABSOLUTE = {
    "geneformer": ("R2(z)", "R2(C1)", "R2(N0[geneformer])"),
    "scgpt": ("R2(scgpt)", "R2(C1[scgpt])", "R2(N0[scgpt])"),
    "stack": ("R2(stack)", "R2(C1[stack])", "R2(N0[stack])"),
    "pca50": ("R2(pca50)", "R2(C1[pca50])", "R2(N0[pca50])"),
}


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(value, places=3):
    quantum = Decimal(1).scaleb(-places)
    return format(Decimal(value).quantize(quantum, rounding=ROUND_HALF_EVEN), "+." + str(places) + "f")


def est(row, key="r2", lo="r2_lo", hi="r2_hi"):
    return f"{num(row[key])} [{num(row[lo])}, {num(row[hi])}]"


def one(rows, **conditions):
    found = [r for r in rows if all(r[k] == v for k, v in conditions.items())]
    if len(found) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(found)}")
    return found[0]


def predictor_sensitivity(vcc):
    header = ["Minimum cells", "Input", "Predictor", "All cells", "Fixed count", "Paired drop", "Analysis status"]
    body = []
    for cell, minimum in CELLS:
        heads = HEADS if cell == "G6" else HEADS[:1]
        for key, name in INPUTS:
            for head, head_name in heads:
                status = "Post hoc / planned / post hoc" if key == "geneformer" else "Planned (later protocol)"
                body.append([
                    minimum, name, head_name,
                    est(one(vcc, encoder=key, cell=cell, head=head, quantity="e - C1", label="existing")),
                    est(one(vcc, encoder=key, cell=cell, head=head, quantity="e - C1", label="rebuilt")),
                    est(one(vcc, encoder=key, cell=cell, head=head, quantity="P-d")),
                    status,
                ])
        for head, head_name in HEADS:
            row = one(vcc, cell=cell, head=head, quantity="R2(B0)", label="existing")
            body.append([minimum, "Cell count only", head_name, est(row), "", "", "Post hoc."])
    return header, body


def absolute_r2(vcc):
    header = ["Minimum cells", "Test groups", "Test genes", "Input", "Score",
              "Input itself", "Gaussian features of the same width", "Count-matched control cells"]
    body = []
    for cell, minimum in CELLS:
        for key, name in INPUTS:
            own, gaussian, matched = ABSOLUTE[key]
            for label, score in SCORES:
                first = one(vcc, encoder=key, cell=cell, head="forest", quantity=own, label=label)
                body.append([
                    minimum, first["n_test_rows"], first["n_test_genes"], name, score, est(first),
                    est(one(vcc, encoder=key, cell=cell, head="forest", quantity=gaussian, label=label)),
                    est(one(vcc, encoder=key, cell=cell, head="forest", quantity=matched, label=label)),
                ])
        for label, score in SCORES:
            row = one(vcc, cell=cell, head="forest", quantity="R2(B0)", label=label)
            body.append([minimum, row["n_test_rows"], row["n_test_genes"], "Cell count only", score, est(row), "", ""])
    return header, body


def crossed_intervals(crossed):
    names = dict(INPUTS)
    quantity = {"e - C1": "Input minus Gaussian features", "P-d": "Paired drop",
                "N0 - C1": "Count-matched control cells minus Gaussian features"}
    score = {"existing": "All cells", "rebuilt": "Fixed count", "existing minus rebuilt": "All cells minus fixed count"}
    header = ["Input", "Quantity", "Score", "Estimate", "Interval over target genes",
              "Interval over batches", "Crossed interval (genes and batches)"]
    body = []
    for row in crossed:
        body.append([
            names.get(row["encoder"], row["encoder"]), quantity.get(row["quantity"], row["quantity"]),
            score.get(row["label"], row["label"]), num(row["r2"]),
            f"[{num(row['gene_lo'])}, {num(row['gene_hi'])}]",
            f"[{num(row['batch_lo'])}, {num(row['batch_hi'])}]",
            f"[{num(row['crossed_lo'])}, {num(row['crossed_hi'])}]",
        ])
    return header, body


def split_half(halves):
    header = ["Setting", "Dataset", "Perturbed cells per half", "Half definition", "Groups or constructs",
              "Target genes", "Score agreement between halves", "Out-of-fold R2 of the five features",
              "Residual agreement [interval]", "Range over fold seeds", "Share explained"]
    body = []
    for row in halves:
        r2 = (Decimal(row["oof_r2_half_A"]) + Decimal(row["oof_r2_half_B"])) / 2
        body.append([
            row["cell"], row["side"], row["level"], row["version"], row["rows"], row["target_genes"],
            num(row["half_label_spearman"]), num(r2),
            f"{num(row['residual_reliability'])} [{num(row['interval_lo'])}, {num(row['interval_hi'])}]",
            f"{num(row['fold_seed_min'])} to {num(row['fold_seed_max'])}",
            num(row["replicable_share_explained"], 4),
        ])
    return header, body


def to_tex(header, body):
    def cell(text):
        text = str(text)
        for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("-", "$-$")):
            text = text.replace(old, new)
        return text
    lines = [r"\begin{tabular}{" + "l" * len(header) + "}", r"\toprule",
             " & ".join(cell(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def write(out, name, table, inputs, repo):
    header, body = table
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)
    (out / f"{name}.tex").write_text(to_tex(header, body), encoding="utf-8")
    record = {"table": name, "rows": len(body), "inputs": {p: digest(repo / p) for p in inputs},
              "rounding": "three decimals, round half to even", "fits": "none"}
    (out / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"{name}: {len(body)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vcc = read(args.repo / VCC)
    write(args.out, "vcc_predictor_sensitivity", predictor_sensitivity(vcc), [VCC], args.repo)
    write(args.out, "vcc_absolute_r2", absolute_r2(vcc), [VCC], args.repo)
    write(args.out, "vcc_crossed_intervals", crossed_intervals(read(args.repo / CROSSED)), [CROSSED], args.repo)
    write(args.out, "split_half_agreement", split_half(read(args.repo / HALVES)), [HALVES], args.repo)


if __name__ == "__main__":
    main()
