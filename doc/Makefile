# Minimal makefile for Sphinx documentation
#

SHELL          = /bin/bash
HAS_BASH       = YES
ifeq (,$(wildcard $(SHELL)))
OSHELL         := $(SHELL)
override SHELL = /bin/sh
HAS_BASH       = NO
endif

BUILDDIR       = ${CURDIR}
RSTDIR         = $(BUILDDIR)/src
VENV           = $(BUILDDIR)/venv
SPHINXCONFIG   = $(BUILDDIR)/conf

PYTHON         = $(word 3,$(shell type python3))
HAS_PYTHON3    = NO

ifeq ($(shell type python3 >/dev/null 2>&1; echo $$?), 0)
HAS_PYTHON3    = YES
endif

# ------------------------------------------

help:
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html          create HTML pages in html dir"
	@echo "  clean         remove all intermediate files"
	@echo "  clean-all     reset the entire build environment"

# ------------------------------------------

$(VENV):
	@if [ "$(HAS_BASH)" == "NO" ] ; then echo "bash was not found at $(OSHELL)! Please use: $(MAKE) SHELL=/path/to/bash" 1>&2; exit 1; fi
	@if [ "$(HAS_PYTHON3)" == "NO" ] ; then echo "python3 was not found! Please see README for further instructions" 1>&2; exit 1; fi
	@( \
		$(PYTHON) -m venv $(VENV); \
		. $(VENV)/bin/activate; \
		pip $(PIP_OPTIONS) install --upgrade pip; \
		pip $(PIP_OPTIONS) install --upgrade wheel; \
		pip $(PIP_OPTIONS) install -r $(BUILDDIR)/conf/requirements.txt; \
		deactivate;\
	)

# ------------------------------------------

EXAMPLES_RST:
	@(cd conf; python examples_rst.py)

# ------------------------------------------

#html: $(VENV)
# . $(VENV)/bin/activate ;
# 		deactivate ;\

html: #EXAMPLES_RST
	@(\
		env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build -E $(SPHINXEXTRA) -b html -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) html ;\
		env PYTHONWARNINGS= PYTHONDONTWRITEBYTECODE=1 \
		sphinx-build $(SPHINXEXTRA) -b html -c $(SPHINXCONFIG) -d $(BUILDDIR)/doctrees $(RSTDIR) html ;\
		echo "############################################" ;\
	)
	@rm -rf html/_sources
	@rm -rf html/USER
	@rm -rf html/JPG
	@rm -rf html/PDF/.[sg]*
	@echo "Build finished. The HTML pages are in doc/html."

# ------------------------------------------

clean-all: clean
	rm -rf $(BUILDDIR)/venv

clean:
	rm -rf $(BUILDDIR)/doctrees $(BUILDDIR)/html $(BUILDDIR)/fasthtml

.PHONY: help Makefile clean-all clean html fasthtml


