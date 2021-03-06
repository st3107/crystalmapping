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


def test_plot_real_aspect():
    """Test plot_real_aspect on a zero image."""
    data = xr.DataArray(np.zeros((5, 5)))
    data[2, 2] = 1
    utils.plot_real_aspect(data)
    # plt.show()
    plt.clf()


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
    # plt.show(block=False)
    plt.clf()
    # test calculation of the light and dark
    c.calc_dark_and_light_from_frames_arr()
    assert np.array_equal(c.dark, np.squeeze(dark_image))
    assert np.array_equal(c.light, np.squeeze(light_image))
    # test the visualization of light and dark
    c.show_dark()
    # plt.show(block=False)
    plt.clf()
    c.show_light()
    # plt.show(block=False)
    plt.clf()
    # test the peak tracking
    c.calc_peaks_from_dk_sub_frame(2, invert=True)
    # test the window drawing
    c.calc_windows_from_peaks(4, 2)
    # test the visualization of windows
    c.show_windows()
    # plt.show(block=False)
    plt.clf()
    # test the calculation of the intensity
    c.calc_intensity_in_windows()
    c.reshape_intensity()
    # test the visualization of the intensity
    c.show_intensity()
    # plt.show(block=False)
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
    # plt.show(block=False)
    plt.clf()
    # check the frame
    c.show_windows()
    # plt.show(block=False)
    plt.clf()
    # check the intensity
    c.show_intensity()
    # plt.show(block=False)
    plt.clf()
    # check the ds
    print(ds)


def test_load_structure():
    cm = utils.CrystalMapper()
    cm.load_structure(CIF_FILE)


def test_load_ai():
    cm = utils.CrystalMapper()
    cm.load_ai(PONI_FILE)
