DMTN-075.pdf: *.tex
	latexmk -bibtex -pdf -f DMTN-075.tex -halt-on-error
