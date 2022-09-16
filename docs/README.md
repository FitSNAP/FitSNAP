Here we're documenting code with Sphinx, following the rules/procedures locatedat https://www.sphinx-doc.org/en/master/usage/quickstart.html.

First we must install Sphinx, which may be done in `pip` with:

    pip install sphinx

This allows us to do the `sphinx-quickstart` command and create the Makefile, `conf.py` file, and overall layout for docs.

To contribute to the docs, first edit the `.rst` files in the `source` directory.

Then build the HTML files with `make html` in this directory.

This creates HTML files in `build/html` that may be viewed for testing purposes.

To clean the HTML files and start anew, simply do `make clean` in this directory.
