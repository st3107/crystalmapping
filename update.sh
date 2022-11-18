set -e
env="$1"
git pull
conda install -n "$env" -c conda-forge -c defaults --file ./requirements-run.txt --yes
conda run -n "$env" pip install -e .
echo "Package has been successfully updated."
