all: simp.pdf

simp.pdf: simp.tex simp.bib cartoon.pdf domainnotation.pdf
	pdflatex simp
	bibtex simp
	pdflatex simp
	pdflatex simp

cartoon.pdf domainnotation.pdf: cartoon.py
	python3 cartoon.py
	for X in cartoon.pdf domainnotation.pdf currenttime.pdf referencedomain.pdf nexttime.pdf; do \
	     (pdfcrop $$X $$X); \
	done

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.synctex.gz *.bbl

