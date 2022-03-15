import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pkg_resources import resource_filename

import crystalmapping.utils as utils


#@pytest.mark.skip
def test_locally():
    """A local test. Skip in CI."""
    DATA_FILE = "/Users/sst/project/analysis/st_crystalmapping/notebooks/data/CG_0046_full_range_grid_scan_90_degree_40_p_20_w.nc"
    CIF_FILE_ = "/Users/sst/project/analysis/st_crystalmapping/notebooks/data/tio2_rutile.cif"
    PONI_FILE_ = "/Users/sst/project/analysis/st_crystalmapping/notebooks/data/CeO2_0.25x0.25_beam.poni"
    cm = utils.CrystalMapper()
    raw_data = xr.open_dataset(DATA_FILE)
    cm.windows = raw_data[["y", "dy", "x", "dx"]].to_dataframe()
    # convert unit to A
    cm.load_structure(CIF_FILE_)
    cm.load_ai(PONI_FILE_)
    # prepare the hkls
    cm.assign_q_values()
    cm.assign_d_values()
    cm.calc_hkls(0.1)
    # index peaks in a group
    GROUP4 = [15, 49, 51]
    hkl01 = np.array([1., -2., -3.])
    hkl21 = np.array([-1., 2., 1.])
    hkl02 = np.array([2., 1., 3.])
    hkl22 = np.array([2., -1., -1.])
    cm._set_us_for_peaks(GROUP4[0], GROUP4[2])
    res1 = cm._index_others(hkl01, hkl21, [GROUP4[1]])
    res2 = cm._index_others(hkl02, hkl22, [GROUP4[1]])
    hkl_1 = res1["hkls"].data[2]
    hkl_2 = res2["hkls"].data[2]
    v1 = cm.ubmatrix.reci_to_cart(hkl_1)
    v2 = cm.ubmatrix.reci_to_cart(hkl_2)
    # ||v|| = Q
    Q = cm.windows["Q"].loc[GROUP4[1]]
    assert abs(np.linalg.norm(v1) - Q) < 1e-6
    assert abs(np.linalg.norm(v2) - Q) < 1e-6
