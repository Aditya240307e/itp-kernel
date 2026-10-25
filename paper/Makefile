# paper/Makefile -- build main.pdf.   make            (needs latexmk, or `make plain` with pdflatex + bibtex)
#                                     make clean
# The tables, figures and numbers come from paper/generated/ (scripts/07_report.sh); without them the PDF still builds, with a box where each is missing and ??name where a number is missing.
all: main.pdf

main.pdf: main.tex main.bib $(wildcard generated/*.tex) $(wildcard generated/tables/*.tex) $(wildcard generated/figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

plain:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C 2>/dev/null || true
	rm -f main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk main.pdf

.PHONY: all plain clean
