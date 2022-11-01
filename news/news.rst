**Added:**

* Add `CrystalMapper` (an important update). It calculates crystal maps and rocking curves using the diffraction data.

* Add `PeakIndexer` (an important new functionality). It guesses orientation of the grains and deduces the miller indexes of Bragg peaks based on their positions on detectors.

**Changed:**

* `PeakIndexer` can process multiple datasets from different orientated samples together.

* `PeakIndexer` can index other Bragg peaks using the U matrix from the previous results.

* Improve the API. Users tune configuration in the `Config` instance instead of the attributes.

* Improve the visualization. Visualization is more automated. The code chooses the optimal plot type, size and scale of figures for the users.

**Deprecated:**

* <news item>

**Removed:**

* Remove old `Calculator`. The `CrystalMapper` will take over its functionality.

**Fixed:**

* <news item>

**Security:**

* <news item>
