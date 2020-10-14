all: simp.pdf

_cartoons := cartoon.pdf domainnotation.pdf currenttime.pdf referencedomain.pdf nexttime.pdf
cartoons := $(addprefix figs/, $(_cartoons))

figures := $(cartoons) figs/coarsespy.pdf

simp.pdf: simp.tex simp.bib $(figures)
	pdflatex simp
	bibtex simp
	pdflatex simp
	pdflatex simp

$(cartoons): figs/cartoon.py
	(cd figs/ && python3 cartoon.py)
	for X in $(cartoons); do \
	     (pdfcrop $$X $$X); \
	done

.PHONY: clean

clean:
	@rm -f $(cartoons)
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.synctex.gz *.bbl

