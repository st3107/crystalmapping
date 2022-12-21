================
rever Change Log
================

.. current developments

v0.3.0
====================



v0.2.0
====================

**Added:**

* Add `CrystalMapper` (an important update). It calculates crystal maps and rocking curves using the diffraction data.

* Add `PeakIndexer` (an important new functionality). It guesses orientation of the grains and deduces the miller indexes of Bragg peaks based on their positions on detectors.

**Changed:**

* `PeakIndexer` can process multiple datasets from different orientated samples together.

* `PeakIndexer` can index other Bragg peaks using the U matrix from the previous results.

* Improve the API. Users tune configuration in the `Config` instance instead of the attributes.

* Improve the visualization. Visualization is more automated. The code chooses the optimal plot type, size and scale of figures for the users.

**Removed:**

* Remove old `Calculator`. The `CrystalMapper` will take over its functionality.



v0.1.1
====================

**Added:**

* Add a method to run the whole process from raw data to crystal map in one run.

* Add a method to find the hkl for a Bragg peak in a range of Q value.

* Add the module `ubmatrix` for the calculation of the U and B matrix.

* Add visualization functions for the crystal maps.

**Changed:**

* Rename the `Calculator` in `crystalmapping.utils` by `CrystalMapper`.

**Removed:**

* Remove the tests for the servers.

* Remove the large image files for the tests.

**Fixed:**

* Fix the several minor reported bugs in visualization.



v0.1.0
====================



v0.0.1
====================

**Added:**

* Add `crystalmapping.plans` for the grid scan and fly scan experiments at NSLS-II.

* Add `crystalmapping.servers` for the automated streaming light and dark frame calculation.

* Add `crystalmapping.utils` for the batch data processing using the new algorithm.

* Add `crystalmapping.callbacks` for the streaming data processing using old CJ's algorithm.


