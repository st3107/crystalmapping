from pathlib import PurePath

from pkg_resources import resource_filename

_DATA_DIR = PurePath(resource_filename("crystalmapping", "data"))
CEO2_PONI_FILE = _DATA_DIR.joinpath("CeO2_0.25x0.25_beam.poni")
CRYSTAL_MAPS_FILE_90_DEG = _DATA_DIR.joinpath(
    "CG_0046_full_range_grid_scan_90_degree_40_p_20_w.nc"
)
CRYSTAL_MAPS_FILE_0_DEG = _DATA_DIR.joinpath(
    "CG_0046_full_range_grid_scan_0_degree_40_p_20_w.nc"
)
FAKE_IMAGE_FILE = _DATA_DIR.joinpath("image.png")
NI_CIF_FILE = _DATA_DIR.joinpath("Ni.cif")
NI_PONI_FILE = _DATA_DIR.joinpath("Ni.poni")
TIO2_CIF_FILE = _DATA_DIR.joinpath("tio2_rutile.cif")
