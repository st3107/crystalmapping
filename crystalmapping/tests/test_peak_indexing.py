from crystalmapping.peakindexer import PeakIndexer, IndexerConfig
from crystalmapping.datafiles import CEO2_PONI_FILE, CRYSTAL_MAPS_FILE, TIO2_CIF_FILE


def test_indexing_real_data(tmpdir):
    GRPOUP1 = [16, 59, 37]
    config = IndexerConfig()
    pi = PeakIndexer(config)
    pi.load(
        str(CRYSTAL_MAPS_FILE),
        str(CEO2_PONI_FILE),
        str(TIO2_CIF_FILE)
    )
    pi.guess_miller_index(GRPOUP1)
    pi.show()
    return
