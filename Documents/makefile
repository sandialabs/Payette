STRICTFLAGS = -halt-on-error -file-line-error
LFILE = Payette_users_and_devel_guide

all: $(LFILE).tex
	@ echo "\n\n===== pdflatex run 1\n"
	pdflatex $(STRICTFLAGS) $(LFILE).tex
	#@ echo "\n\n===== bibtex\n"
	#bibtex $(LFILE)
	@ echo "\n\n===== pdflatex run 2\n"
	pdflatex $(STRICTFLAGS) $(LFILE).tex
	@ echo "\n\n===== pdflatex run 3\n"
	pdflatex $(STRICTFLAGS) $(LFILE).tex
	@ echo "\n\n===== cleaning up\n"
	make clean

diag: $(LFILE).tex
	pdflatex $(LFILE).tex
	pdflatex $(LATEXFLAGS) $(LFILE).tex
	pdflatex $(LATEXFLAGS) $(LFILE).tex

bib: $(LFILE).bib 
	./MakeBibReference.py $(LFILE).bib >> DeleteMe.tex
	pdflatex DeleteMe.tex
	bibtex DeleteMe
	pdflatex DeleteMe.tex
	pdflatex DeleteMe.tex
	mv DeleteMe.pdf BibliographyLookup.pdf
	rm -f DeleteMe.*

#
# Remove intermediate files (not really needed)
#
clean:
	rm $(LFILE).aux
	rm $(LFILE).log
