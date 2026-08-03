from __future__ import annotations

from importlib.metadata import version as distribution_version
from pathlib import Path
import sys

DOCS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_ROOT.parent

sys.path.insert(0, str(DOCS_ROOT / "_ext"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "Mimodium"
author = "Leonard Schulz"
release = distribution_version("mimodium")
html_title = "Mimodium"
html_short_title = "Mimodium"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "mimodium_semantic_types",
]

exclude_patterns = ["_build", ".ipynb_checkpoints", "tutorial_sources"]
master_doc = "index"

sphinx_gallery_conf = {
    "examples_dirs": "tutorial_sources/hello_world",
    "gallery_dirs": "tutorials/hello_world",
    "download_all_examples": False,
}

html_theme = "pydata_sphinx_theme"
html_logo = "_static/brand/mimodium-mark.svg"
html_theme_options = {
    "github_url": "https://github.com/LPSchulz/mimodium",
    "logo": {
        "image_light": "_static/brand/mimodium-mark.svg",
        "image_dark": "_static/brand/mimodium-mark-dark.png",
        "text": "Mimodium",
    },
    "show_toc_level": 2,
}
html_sidebars = {
    "semantic_types": ["sidebar-collapse", "sidebar-nav-bs", "page-toc"],
}
html_static_path = ["_static"]
html_css_files = ["mimodium.css"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "special-members": "__call__",
}
autodoc_class_signature = "mixed"
autodoc_typehints = "signature"
autodoc_inherit_docstrings = False
python_use_unqualified_type_names = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

mimodium_type_package_root = str(PROJECT_ROOT / "src" / "mimodium")
mimodium_type_package_name = "mimodium"
mimodium_type_package_order = [
    "scenario",
    "propagation",
    "algorithms",
    "evaluation",
    "visualization",
]
