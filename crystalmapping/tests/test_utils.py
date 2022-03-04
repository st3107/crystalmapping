import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pkg_resources import resource_filename

import crystalmapping.utils as utils

plt.ioff()
IMAGE_FILE = resource_filename("crystalmapping", "data/image.png")
CIF_FILE = resource_filename("crystalmapping", "data/Ni.cif")
PONI_FILE = resource_filename("crystalmapping", "data/Ni.poni")


def test_reshape():
    """Test reshape in a snaking axes case."""
    ds = xr.Dataset({"array_data": [0, 0, 0, 1, 0, 0, 0, 0, 0]})
    ds.attrs["shape"] = (3, 3)
    ds.attrs["extents"] = [(-1, 1), (-1, 1)]
    ds.attrs["snaking"] = [False, True]
    res = utils.reshape(ds, "array_data")
    expected = np.zeros((3, 3))
    expected[1, 2] = 1
    assert np.array_equal(res.values, expected)


def test_plot_real_aspect():
    """Test plot_real_aspect on a zero image."""
    data = xr.DataArray(np.zeros((5, 5)))
    data[2, 2] = 1
    utils.plot_real_aspect(data)
    # plt.show()
    plt.clf()


@pytest.mark.skip
def test_annotate_peaks():
    """Test annotate_peaks in a toy case."""
    df = pd.DataFrame(
        {"y": [2], "x": [2], "mass": [1], "size": [1], "ecc": [1], "signal": [1], "raw_mass": [1], "frame": [0]}
    )
    image = xr.DataArray(np.zeros((5, 5)))
    image[2, 2] = 1
    utils.annotate_peaks(df, image)
    # plt.show()
    plt.clf()


@pytest.mark.skip
def test_create_atlas():
    """Test create_atlas by plotting the figure out. Test assign_Q_to_atlas"""
    df = pd.DataFrame(
        {"y": [1, 2], "x": [1, 2], "mass": [1, 2], "size": [1, 2], "ecc": [1, 2], "signal": [1, 2],
         "raw_mass": [1, 2], "frame": [1, 3], "particle": [0, 1]}
    )
    df.attrs["shape"] = (2, 2)
    df.attrs["extents"] = [(-3, 3), (-1, 1)]
    df.attrs["snaking"] = (False, True)
    ds = utils.create_atlas(df)
    # print(ds)
    facet = utils.plot_grain_maps(ds)
    facet.fig.set_size_inches(4, 4)
    # facet.fig.show()
    plt.clf()
    # test
    ai = utils.AzimuthalIntegrator(detector="Perkin", wavelength=2 * np.pi)
    ds2 = utils.assign_Q_to_atlas(ds, ai)
    print(ds2)


def test_map_to_Q():
    """Test map_to_Q_vector."""
    # test numpy
    d1 = np.array([0, 1])
    d2 = np.array([0, 1])
    ai = utils.AzimuthalIntegrator(detector="Perkin", wavelength=2 * np.pi)
    q = utils.pixel_to_Q(d1, d2, ai)
    assert q.shape == (2,)


def test_Calculator_step_by_step():
    c = utils.CrystalMapper()
    # load test data
    light_image: np.ndarray = plt.imread(IMAGE_FILE)
    light_image = np.expand_dims(light_image, 0)
    dark_image: np.ndarray = np.zeros_like(light_image)
    # give the data to the calculator
    c.frames_arr = xr.DataArray([light_image, light_image, dark_image, dark_image])
    c.metadata = {"shape": [2, 2], "extents": [(0, 1), (0, 1)], "snaking": (False, False)}
    c.ai = utils.AzimuthalIntegrator(
        detector="dexela2923",
        wavelength=0.168 * 1e-9,
        dist=0.01
    )
    c.cell = utils.Cell(a=50, b=50, c=50)
    # test the show frames
    c.show_frame(0)
    plt.show(block=False)
    plt.clf()
    # test calculation of the light and dark
    c.calc_dark_and_light_from_frames_arr()
    assert np.array_equal(c.dark, np.squeeze(dark_image))
    assert np.array_equal(c.light, np.squeeze(light_image))
    # test the visualization of light and dark
    c.show_dark()
    plt.show(block=False)
    plt.clf()
    c.show_light()
    plt.show(block=False)
    plt.clf()
    # test the peak tracking
    c.calc_peaks_from_dk_sub_frame(2, invert=True)
    # test the window drawing
    c.calc_windows_from_peaks(4, 2)
    # test the visualization of windows
    c.show_windows()
    plt.show(block=False)
    plt.clf()
    # test the calculation of the intensity
    c.calc_intensity_in_windows()
    c.reshape_intensity()
    # test the visualization of the intensity
    c.show_intensity()
    plt.show(block=False)
    plt.clf()
    # test the calculation of the coordinates
    c.calc_coords()
    # test the hkl indexing
    c.calc_hkls_in_a_range(0.99, 1.01)
    # test the hkl bounds
    c.calc_hkls(0.2)
    # test export dataset
    ds = c.to_dataset()
    print(ds)


def test_index_peaks_in_one_grain():
    c = utils.CrystalMapper()
    # load test data
    light_image: np.ndarray = plt.imread(IMAGE_FILE)
    light_image = np.expand_dims(light_image, 0)
    dark_image: np.ndarray = np.zeros_like(light_image)
    # give the data to the calculator
    c.frames_arr = xr.DataArray([light_image, light_image, dark_image, dark_image])
    c.metadata = {"shape": [2, 2], "extents": [(0, 1), (0, 1)], "snaking": (False, False)}
    ai = utils.AzimuthalIntegrator(
        detector="dexela2923",
        wavelength=0.168 * 1e-9,
        dist=0.01
    )
    c.ai = ai
    c.ubmatrix.geo = ai
    c.cell = utils.Cell(a=50, b=50, c=50)
    c.ubmatrix.lat = utils.Lattice(a=50, b=50, c=50, alpha=90, beta=90, gamma=90)
    # find peaks
    c.calc_dark_and_light_from_frames_arr()
    c.calc_peaks_from_dk_sub_frame(2, invert=True)
    # test the window drawing
    c.calc_windows_from_peaks(4, 2)
    # run the indexing of peaks
    c.calc_hkls(0.0001)
    returned = c.index_peaks_in_one_grain(c.windows.index[:3])
    assert np.all(np.logical_not(np.isnan(returned["hkls"])))


def test_Calculator_auto_processing_and_reload():
    c = utils.CrystalMapper()
    # load test data
    light_image: np.ndarray = plt.imread(IMAGE_FILE)
    light_image = np.expand_dims(light_image, 0)
    dark_image: np.ndarray = np.zeros_like(light_image)
    # give the data to the calculator
    c.frames_arr = xr.DataArray([light_image, light_image, dark_image, dark_image])
    c.metadata = {"shape": [2, 2], "extents": [(0, 1), (0, 1)], "snaking": (False, False)}
    c.ai = utils.AzimuthalIntegrator(
        detector="dexela2923",
        wavelength=0.168 * 1e-9,
        dist=0.01
    )
    c.cell = utils.Cell(a=5, b=5, c=5)
    # run auto processing
    c.auto_process(4, 2, 2, invert=True)
    # test reloading
    ds = c.to_dataset()
    c.load_dataset(ds)
    ds2 = c.to_dataset()
    assert ds2.equals(ds)
    # check the frame
    c.show_frame(0)
    plt.show(block=False)
    plt.clf()
    # check the frame
    c.show_windows()
    plt.show(block=False)
    plt.clf()
    # check the intensity
    c.show_intensity()
    plt.show(block=False)
    plt.clf()
    # check the ds
    print(ds)


def test_load_structure():
    cm = utils.CrystalMapper()
    cm.load_structure(CIF_FILE)


def test_load_ai():
    cm = utils.CrystalMapper()
    cm.load_ai(PONI_FILE)


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
