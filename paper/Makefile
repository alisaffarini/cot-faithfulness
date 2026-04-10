# CoT Faithfulness Paper - Build
# Requires: pdflatex, bibtex (or latexmk)
# Note: uses inline \begin{thebibliography}, so no separate .bib file needed.
# If you switch to a .bib file, uncomment the bibtex lines.

MAIN = main
TEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

# If using .bib file instead of inline thebibliography:
# $(MAIN).pdf: $(MAIN).tex references.bib
# 	$(TEX) $(MAIN)
# 	$(BIBTEX) $(MAIN)
# 	$(TEX) $(MAIN)
# 	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz
