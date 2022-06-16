==============
crystalmapping
==============

Obtain the distribution of grains inside a crystal from a series of diffraction images using automated peak
tracking.

* Free software: 3-clause BSD license
* Documentation: https://st3107.github.io/crystalmapping.


Install
=======

Run the bash script in a `bash` shell.

``bash install.sh``

Update
======

Run the following commands to update.

``conda actiavte <the name of the env that you created>```

``git pull origin main``

``python -m pip install .``

If some of the dependencies are out of date,

``conda install -c conda-forge --file requirements-dev.txt --yes``

Examples
========

Please ask Songsheng to give the access rights to this google folder.

https://drive.google.com/drive/folders/13DJwlidULEW7Zgaubd3fiIlQeL7OiYZF?usp=sharing

Then, there is two example jupter notebooks in the `notebooks` folder that you can run in JupyterLab.

Below is the notebook to show how to get crystal maps and table of peaks information from the diffraction data.

01_analysis_example_code.ipynb

Below is the notebook to show how to guess the indexes of the peaks from the same grain using the results from the crystal mappings in the former step.

08_how_to_run_peak_indexing.ipynb
