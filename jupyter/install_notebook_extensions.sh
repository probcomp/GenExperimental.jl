#!/bin/sh
set -e
set -u

# install Gen notebook extension
jupyter nbextension install gen_notebook_extension/ --user
jupyter nbextension enable gen_notebook_extension/main --user

# install D3 as a notebook extension
mkdir -p ~/.local/share/jupyter/nbextensions/d3
cp resources/d3.min.js ~/.local/share/jupyter/nbextensions/d3
jupyter nbextension enable d3/d3.min --user
