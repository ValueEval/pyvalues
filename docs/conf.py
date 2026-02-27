import os
import sys
import importlib.metadata

# Add your source code path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

meta = importlib.metadata.metadata("pyvalues") 

project = meta["Name"]
author = meta["Author"]
release = meta["Version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
]

html_theme = "furo"

# Optional: configure version selector
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"
