============
Installation
============

Installation from conda
-----------------------

At the command line::

    conda install -c defaults -c nsls2forge -c conda-forge -c st3107 crystalmapping


Installation from pip
---------------------

At the command line::

    curl -LJO https://github.com/st3107/tomology/blob/main/run-env.yaml
    conda env create -f run-env.yaml -n crystalmapping
    conda activate crystalmapping
    pip install crystalmapping


Installation from source
------------------------

At the command line::

    curl -s -L https://raw.githubusercontent.com/st3107/crystalmapping/main/install.sh -o install.sh; bash install.sh

