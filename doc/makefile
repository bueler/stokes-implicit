all: simp.pdf

simp.pdf: simp.tex simp.bib cartoon.pdf domainnotation.pdf
	pdflatex simp
	bibtex simp
	pdflatex simp
	pdflatex simp

cartoon.pdf domainnotation.pdf: cartoon.py
	python3 cartoon.py
	pdfcrop cartoon.pdf cartoon.pdf
	pdfcrop domainnotation.pdf domainnotation.pdf

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.synctex.gz *.bbl

