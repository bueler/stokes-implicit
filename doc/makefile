all: simp.pdf

simp.pdf: simp.tex simp.bib
	pdflatex simp
	bibtex simp
	pdflatex simp
	pdflatex simp

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.synctex.gz *.bbl

