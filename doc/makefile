all: simp.pdf

figures := cartoon.pdf domainnotation.pdf currenttime.pdf referencedomain.pdf nexttime.pdf

simp.pdf: simp.tex simp.bib $(figures)
	pdflatex simp
	bibtex simp
	pdflatex simp
	pdflatex simp

$(figures): cartoon.py
	python3 cartoon.py
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

.PHONY: clean

clean:
	@rm -f $(figures)
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.synctex.gz *.bbl

