# Rebuild the tables and figures of the article from the saved statistics.
# Run from the repository root with the environment of bioinf2027/requirements.txt.
PYTHON ?= python
F = bioinf2027/scripts/figures

.PHONY: test tables figures check

test:
	$(PYTHON) -m pytest

tables:
	$(PYTHON) $(F)/table1_datasets.py
	$(PYTHON) $(F)/table2_main_estimates.py
	$(PYTHON) $(F)/tableS_a_d2_family.py
	$(PYTHON) $(F)/tableS_b_e4_family.py
	$(PYTHON) $(F)/tableS_c_external_values.py
	$(PYTHON) $(F)/tableS_d_e2d.py
	$(PYTHON) $(F)/tableS_e_supplement_detail.py
	$(PYTHON) $(F)/tableS_j_closing.py
	$(PYTHON) $(F)/make_extra_tables.py --repo . --out bioinf2027/tables

figures:
	$(PYTHON) $(F)/fig2_endpoint_cell_number.py --version main
	$(PYTHON) $(F)/fig3_vcc_difference.py
	$(PYTHON) $(F)/fig3_vcc_difference.py --cell G5
	$(PYTHON) $(F)/fig3_vcc_difference.py --metric spearman
	$(PYTHON) $(F)/fig3_vcc_difference.py --pca768
	$(PYTHON) $(F)/fig4_between_screens.py
	$(PYTHON) $(F)/fig4_between_screens.py --part c2 --width 86
	$(PYTHON) $(F)/fig5_identity_task.py --version a
	$(PYTHON) $(F)/fig5_identity_task.py --version b
	$(PYTHON) $(F)/figS_a_external_values.py
	$(PYTHON) $(F)/figS_b_identity_supplement.py

# After "make tables figures": every CSV, LaTeX and SVG output must equal its stored checksum.
check:
	$(PYTHON) tools/check_outputs.py
