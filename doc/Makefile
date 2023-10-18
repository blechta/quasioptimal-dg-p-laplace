PROJECT=main

.PHONY:	all clean FORCE

all: $(PROJECT).pdf

$(PROJECT).pdf:

%.pdf: %.tex FORCE
	latexmk -pdf -shell-escape -quiet -f $<

clean:
	latexmk -C -pdf -e '$$cleanup_includes_generated=1; $$bibtex_use=2' -f -quiet $(PROJECT).pdf
