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
    'sphinx.ext.autodoc',      # Document code automatically from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    # 'sphinx.ext.autosummary',   # Uncomment if you want autosummary
    # 'autoapi.extension',        # Uncomment if using sphinx-autoapi
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc options (optional) ----------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'inherited-members': True,
    'show-inheritance': True,
}
autodoc_mock_imports = []

# -- Autosummary options (optional) ------------------------------------------
# autosummary_generate = True

# -- AutoAPI options (optional) ----------------------------------------------
autoapi_type = 'python'
autoapi_dirs = ['../../yrc']  # Adjust to your code directory

# -- Napoleon settings (optional, for Google/NumPy style docstrings) ---------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

