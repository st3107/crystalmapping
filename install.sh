set -e
read -p "Please give a name for the conda environment: " env
echo "Download crystalmapping repo ..."
conda env create -n "$env"
conda install -n "$env" --file ./requirements-run.txt --yes
conda run -n "$env" python3 -m pip install -e .
echo "Package has been successfully installed."
