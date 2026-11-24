"""Table S-h: the encoders' inputs and coverage (the stop of 2026-11-17 06:50, figure and table plan, item 7: "scGPT's gene
cut, Stack's gene list, PCA's features, the coverage of the identity vectors"). Two parts:
- part 1, the cell encoders per screen: the input setting with the line that states it (each checked to appear on that
  line), genes of the screen's file matched, the 2,000 feature genes and target genes matched, the median share of a
  cell's counts in matched genes, and for scGPT the cut to 1,199 genes and the share of counts after it
  (bioinf2027/results/E2/coverage.json, coverage_scgpt_input.json, probe_stack.json, features.json), and for PCA the
  feature genes mapped in each screen (bioinf2027/results/E1/checks.json);
- part 2, the identity vectors of E5 per screen: target genes with a vector, and those covered by all three vectors and
  Bc (bioinf2027/results/E5/identity/*.json; bioinf2027/results/E5/supplement/supplement_lists.json, key `counts`).
Usage, from the repository root: python bioinf2027/scripts/figures/tableS_h_encoder_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

REPO = tablelib.REPO
R = REPO / "bioinf2027" / "results"
COVERAGE, SCGPT_INPUT = R / "E2" / "coverage.json", R / "E2" / "coverage_scgpt_input.json"
STACK, FEATURES = R / "E2" / "probe_stack.json", R / "E2" / "features.json"
CHECKS = R / "E1" / "checks.json"
IDENTITY = {"Geneformer token (768)": R / "E5/identity/geneformer_token.json", "scGPT gene (512)": R / "E5/identity/scgpt_gene.json",
            "GenePT text (1,536)": R / "E5/identity/genept.json"}
LISTS = R / "E5" / "supplement" / "supplement_lists.json"
SCREENS = ("VCC", "RPE1", "K562-essential")
# (encoder, input setting as printed, file and line that state it, text that must appear on that line)
SETTINGS = (
    ("Geneformer V2-104M", "up to 4,096 ranked genes per cell", "bioinf2027/plans/E1_design.md", 565, "up to 4,096 ranked genes"),
    ("scGPT whole-human", "max_length 1,200; a random 1,199 of the expressed vocabulary genes per cell", "bioinf2027/plans/E2.md", 102, "`max_length` 1,200"),
    ("Stack-Large", "its released gene list; sets of 256 cells", "bioinf2027/results/E2/probe_stack.json", None, None),
    ("PCA-50, PCA-768", "VCC control cells' axes on the 2,000 feature genes", "bioinf2027/plans/E2.md", 126, "38,176 cells"),
)


def cited(path: str, line: int, text: str) -> None:
    if text not in (REPO / path).read_text().split("\n")[line - 1]:
        raise SystemExit("%s:%d does not hold %r" % (path, line, text))


def share(v: float) -> str:
    return "%.3f" % v


def main() -> None:
    coverage = json.loads(COVERAGE.read_text())["coverage"]
    scgpt_input = json.loads(SCGPT_INPUT.read_text())["screens"]
    stack = json.loads(STACK.read_text())
    pca = json.loads(FEATURES.read_text())["outputs"]["pca"]
    mapped = {s: json.loads(CHECKS.read_text())["external_x"][s]["features_mapped"] for s in ("RPE1", "K562-essential")}
    if stack["model"]["model_config"]["n_genes"] != 15012 or stack["model"]["set_size"] != 256:
        raise SystemExit("probe_stack.json does not hold the gene list size or set size printed")
    rows = []
    for encoder, setting, path, line, text in SETTINGS:
        if line is not None:
            cited(path, line, text)
        source = "%s:%d" % (path, line) if line else path
        key = {"Geneformer V2-104M": None, "scGPT whole-human": "scgpt", "Stack-Large": "stack", "PCA-50, PCA-768": None}[encoder]
        for screen in SCREENS:
            row = {"Encoder": encoder, "Input": setting, "Screen": screen}
            if key:
                c = coverage[key][screen]
                median_seen = sorted(c["median_share_of_counts_seen"].values())
                row.update({"Genes in the file": "{:,}".format(c["genes_in_file"]),
                            "Matched to the encoder": "{:,}".format(c["genes_matched"]) + (" of %s" % "{:,}".format(coverage[key]["vocabulary_size"])),
                            "Feature genes matched (of 2,000)": "{:,}".format(c["feature_genes_in_file_and_matched"]),
                            "Target genes matched": share(c["share_of_target_genes_matched"]),
                            "Median share of counts in matched genes": "%s to %s" % (share(median_seen[0]), share(median_seen[-1]))})
            if key == "scgpt":
                s = scgpt_input[screen]
                q = s["share_of_counts_in_the_genes_input_after_the_cut"]["quartiles"]
                row.update({"Median expressed vocabulary genes per cell": "{:,}".format(s["median_expressed_vocabulary_genes"]).rstrip("0").rstrip("."),
                            "Share of cells above 1,199": "%.4f" % s["share_above_1199_expressed"],
                            "Median share of counts after the cut [quartiles]": "%s [%s, %s]" % (share(q[1]), share(q[0]), share(q[2]))})
            if encoder.startswith("PCA"):
                row["Feature genes matched (of 2,000)"] = "2,000" if screen == "VCC" else "{:,}".format(mapped[screen])
                row["Variance explained (50; 768)"] = "%s; %s" % (share(pca["explained_by_50"]), share(pca["explained_by_768"]))
            row["Source"] = "; ".join([source] + (["bioinf2027/results/E2/coverage.json"] if key else [])
                                      + (["bioinf2027/results/E2/coverage_scgpt_input.json"] if key == "scgpt" else [])
                                      + (["bioinf2027/results/E2/features.json", "bioinf2027/results/E1/checks.json"] if encoder.startswith("PCA") else []))
            rows.append(row)
    one = pd.DataFrame(rows).fillna("")
    one = one[[c for c in one.columns if c != "Source"] + ["Source"]]
    lists = json.loads(LISTS.read_text())["counts"]
    vectors = {name: json.loads(p.read_text())["coverage"] for name, p in IDENTITY.items()}
    two = []
    for screen in SCREENS:
        row = {"Screen": screen, "Target genes": "{:,}".format(vectors["Geneformer token (768)"][screen]["genes"])}
        for name in IDENTITY:
            if vectors[name][screen]["genes"] != vectors["Geneformer token (768)"][screen]["genes"]:
                raise SystemExit("the identity records count other target genes in %s" % screen)
            row["With a %s vector" % name] = "{:,}".format(vectors[name][screen]["covered"])
        row["Covered by all three and Bc"] = "{:,}".format(lists[screen]["covered"]) if screen in lists else ""
        two.append(row)
    two = pd.DataFrame(two)
    script = Path(__file__).resolve()
    tablelib.save("tableS_h_encoder_inputs", one, [COVERAGE, SCGPT_INPUT, STACK, FEATURES, CHECKS] + [REPO / s[2] for s in SETTINGS if s[3]], script,
                  align="l" * len(one.columns),
                  extra={"part": "1 of 2, the cell encoders; for PCA the external x is mapped to the 2,000 VCC features by symbol, unmapped features zero "
                                 "(bioinf2027/results/E1/checks.json, key external_x); no coverage count of Geneformer's token dictionary over the files is recorded"})
    tablelib.save("tableS_h_identity_coverage", two, list(IDENTITY.values()) + [LISTS], script, align="l" * len(two.columns),
                  extra={"part": "2 of 2, the identity vectors of E5; VCC is not an E5 setting, so its joint coverage is not computed"})


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_h_encoder_inputs.py")
    main()
