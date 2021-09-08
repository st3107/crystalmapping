============
Installation
============

Installation from conda
-----------------------

At the command line::

    conda install -c defaults -c nsls2forge -c conda-forge -c st3107 tomology


Installation from pip
---------------------

At the command line::

    curl -LJO https://github.com/st3107/tomology/blob/main/run-env.yaml
    conda env create -f run-env.yaml -n tomology
    conda activate tomology
    pip install tomology


Installation in development mode
--------------------------------

At the command line::

    git clone https://github.com/st3107/tomology.git
    cd tomology
    conda env create -f dev-env.yaml -n tomology
    conda activate tomology
    pip install -e .

