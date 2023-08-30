Contributing
============

Important points to follow are explained below.

Style Guide
-----------

Please follow the current style conventions in main FitSNAP files such as :code:`fitsnap3lib/fitsnap.py`. 

- Indent size is 4 spaces, and use actual spaces (not tabs); set your tab key to equal 4 spaces.
- Maximum line length is 100 characters. Set a vertical ruler in your editor and try to terminate
  lines once they reach 100 characters. This makes code easier to read without horiztonal scrolling.
- Naming styles: :code:`module_name`, :code:`package_name`, :code:`ClassName`, :code:`method_name`, 
  :code:`ExceptionName`, :code:`function_name`, :code:`GLOBAL_CONSTANT_NAME`, :code:`global_var_name`, 
  :code:`instance_var_name`, :code:`function_parameter_name`, :code:`local_var_name`
- Avoid global variables that are declared outside class methods or attributes, except in necessary 
  circumstances; sometimes it's useful, but you need to make sure that it doesn't affect unrelated 
  FitSNAP uses.
- Prefer imported functions to new classes with state dependence.

For other style advice, consult Google's Python style guide when in doubt:
https://google.github.io/styleguide/pyguide.html

Documenting
-----------

All new features and examples must be documented. If adding a new feature, it should be explain in 
the appropriate section of our docs. For example if adding a new scraper capability, elaborate in 
`Scraper <Run.html#scraper>`__. This is done by editing the RST files :code:`docs/source` and 
building with Sphinx; see more info in the README in the :code:`docs` directory.

Classes and functions should contain docstrings. We use Google style docstrings, see 
:code:`fitsnap3lib/fitsnap.py` for examples.

New examples must be documented in a README in their appropriate directory. Be specific on how to 
run the example. 

More information on Google's style guide for docs: 

https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e 

https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html

Please reach out or raise an issue on GitHub if you want more info about adding 
a new feature.

