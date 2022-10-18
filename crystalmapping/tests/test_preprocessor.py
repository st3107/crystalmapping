from crystalmapping.datafiles import CRYSTAL_MAPS_FILE_0_DEG, CRYSTAL_MAPS_FILE_90_DEG
from crystalmapping.preprocessor import Preprocessor


def test_Preprocessor():
    dc = Preprocessor("grain", ["x", "y"])
    dc.load_data(CRYSTAL_MAPS_FILE_0_DEG, 0., 0., 0.)
    dc.load_data(CRYSTAL_MAPS_FILE_90_DEG, 0., 0., 90.)
    dc.process()
    dc.pop()
    dc.clear()
