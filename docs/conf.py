# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sysplot'
copyright = '2026, Jan Hoegen'
author = 'Jan Hoegen'
release = '2026-02-16'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For docstrings
    'sphinx.ext.viewcode',  # Add source code links
    'sphinx.ext.intersphinx',  # Link to other docs
    "sphinx.ext.mathjax",
    'sphinx_autodoc_typehints',    # shows type hints, but be last entry!
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Extensions Configuration ---------------------------------------------------

# Autodoc behavior
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # "inherited-members": True,
}

# Autosummary settings
autosummary_generate = True

# Napoleon settings
napoleon_numpy_docstring = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = False

# gallery config
sphinx_gallery_conf = {
    'examples_dirs': 'examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_context = {
   "default_mode": "light"
}

html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
}

html_static_path = ['_static']
