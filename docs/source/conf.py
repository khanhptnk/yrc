# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'YRC'
copyright = '2025, Khanh Nguyen, Mohamad Danesh, Alina Trinh, Ben Plaut'
author = 'Khanh Nguyen, Mohamad Danesh, Alina Trinh, Ben Plaut'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',       # Enables AutoAPI
    'sphinx.ext.napoleon',     # For Google/NumPy-style docstrings
    'sphinx.ext.viewcode',     # Show highlighted source code
    'sphinx_copybutton',
]

autoapi_type = 'python'
autoapi_dirs = ['../../yrc']      # Path(s) to your Python source code

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Additional options ------------------------------------------------------

# If you get import errors for C/C++/external modules, you can add them here
# autodoc_mock_imports = []

# -- End of file -------------------------------------------------------------
