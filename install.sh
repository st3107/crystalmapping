#!/bin/bash

set -e
read -p -r "Please give a name for the conda environment: " env
echo "Download crystalmapping repo ..."
git clone https://github.com/st3107/crystalmapping.git
cd crystalmapping
conda env create -n "$env" --yes
conda run -n "$env" install --file ./requirements-run.txt --yes
conda run -n "$env" python3 -m pip install -e .
echo "Package has been successfully installed."
