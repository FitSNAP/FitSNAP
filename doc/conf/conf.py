# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import site
#__file__ = "../fitsnap3lib/"
__file__ = ".."
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, site.getsitepackages()[0])
sys.path.insert(0, __file__)
#print("sys.path=", sys.path)
#print(site.getsitepackages()[0])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FitSNAP'
copyright = '2022-2025, Sandia Corporation'
author = 'Sandia Corporation'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_tabs.tabs',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.youtube',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []
bibtex_bibfiles = ['reaxff.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'lammps_theme'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['../src/_static']
html_theme_path = ['.']
html_style = 'lammps.css'
html_logo = '../images/FitSNAP.png'
html_favicon = 'fs_favicon.ico'
html_theme_options = {
    'logo_only': True,
    'version_selector': False
}
