FitSNAP code is documented with Sphinx, following the rules/procedures located at https://www.sphinx-doc.org/en/master/usage/quickstart.html.

First we must install Sphinx, which may be done in `pip` with:

    python -m pip install sphinx sphinx_rtd_theme sphinxcontrib-napoleon

This allows us to do the `sphinx-quickstart` command and create the Makefile, `conf.py` file, and overall layout for docs.

Then HTML doc files can be built using:

    make html

To contribute to the docs, edit the `.rst` files in the `source` directory, followed by `make html`.

This creates HTML files in `build/html` that may be viewed for testing purposes.

To clean the HTML files and start anew, simply do `make clean` in this directory.
