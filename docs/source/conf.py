# Configuration file for the Sphinx documentation builder.

project = 'YRC'
copyright = '2025, Khanh Nguyen, Mohamad Danesh'
author = 'Khanh Nguyen, Mohamad Danesh'
release = '1.0.0'

extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

autoapi_type = 'python'
autoapi_dirs = ['../../yrc']
autoapi_template_dir = '_templates/autoapi'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_logo = "images/logo.png"
html_favicon = "images/logo.png"
# html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# End of file

