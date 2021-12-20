import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pkg_resources import resource_filename

import crystalmapping.utils as utils

plt.ioff()
IMAGE_FILE = resource_filename("crystalmapping", "data/image.png")
DEXELA_LIGHT_IMAGE_FILE = resource_filename("crystalmapping", "data/dexela_light_image.npy")
DEXELA_DARK_IMAGE_FILE = resource_filename("crystalmapping", "data/dexela_dark_image.npy")


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
    c.calc_hkls(0.99, 1.01)
    # test export dataset
    ds = c.to_dataset()
    print(ds)


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
