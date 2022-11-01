from crystalmapping.datafiles import (
    CEO2_PONI_FILE,
    CRYSTAL_MAPS_FILE_0_DEG,
    CRYSTAL_MAPS_FILE_90_DEG,
    TIO2_CIF_FILE,
)
from crystalmapping.peakindexer import IndexerConfig, PeakIndexer


def test_indexing_merged_data():
    GRPOUP1 = ["1_1", "1_2"]
    config = IndexerConfig()
    pi = PeakIndexer(config)
    pi.load(
        [str(CRYSTAL_MAPS_FILE_0_DEG), str(CRYSTAL_MAPS_FILE_90_DEG)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 90.0)],
        str(CEO2_PONI_FILE),
        str(TIO2_CIF_FILE),
    )
    pi.guess_miller_index(GRPOUP1)
    pi.show(1)
    pi.visualize(0, ["1_1"])
    pi.hist_error(["1_1", "1_2"])
    pi._previous_result = pi._peak_index
    pi.index_peaks_by_U([0])
    return
