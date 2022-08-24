import bisect
import itertools
import math
import pathlib
import typing
from collections import defaultdict

import pyFAI
import yaml
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import trackpy as tp
import xarray as xr
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from xarray.plot import FacetGrid
from pyFAI.calibrant import Cell
from diffpy.structure import loadStructure, Structure, Lattice


from .ubmatrix import UBMatrix

_VERBOSE = 1
COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
HKL = np.ndarray


def _tableize(df):
    if not isinstance(df, pd.DataFrame):
        return
    df_columns = df.columns.tolist()

    def max_len_in_lst(lst):
        return len(sorted(lst, reverse=True, key=len)[0])

    def align_center(st, sz):
        return (
            "{0}{1}{0}".format(" " * (1 + (sz - len(st)) // 2), st)[:sz]
            if len(st) < sz
            else st
        )

    def align_right(st, sz):
        return "{0}{1} ".format(" " * (sz - len(st) - 1), st) if len(st) < sz else st

    max_col_len = max_len_in_lst(df_columns)
    max_val_len_for_col = dict(
        [
            (col, max_len_in_lst(df.iloc[:, idx].astype("str")))
            for idx, col in enumerate(df_columns)
        ]
    )
    col_sizes = dict(
        [
            (col, 2 + max(max_val_len_for_col.get(col, 0), max_col_len))
            for col in df_columns
        ]
    )

    def build_hline(row):
        return "+".join(["-" * col_sizes[col] for col in row]).join(["+", "+"])

    def build_data(row, align):
        return "|".join(
            [align(str(val), col_sizes[df_columns[idx]]) for idx, val in enumerate(row)]
        ).join(["|", "|"])

    hline = build_hline(df_columns)
    out = [hline, build_data(df_columns, align_center), hline]
    for _, row in df.iterrows():
        out.append(build_data(row.tolist(), align_right))
    out.append(hline)
    return "\n".join(out)


def set_verbose(level: int) -> None:
    global _VERBOSE
    _VERBOSE = level


def _my_print(*args, **kwargs) -> None:
    if _VERBOSE > 0:
        print(*args, **kwargs)


def plot_real_aspect(
    xarr: xr.DataArray, *args, alpha: float = 1.6, **kwargs
) -> xr.plot.FacetGrid:
    """Visualize two dimensional arr as a color map. The color ranges from median - alpha * std to median +
    alpha * std."""
    facet = xarr.plot(*args, **kwargs, **_get_vlim(xarr, alpha))
    _set_real_aspect(facet.axes)
    return facet


def _get_vlim(xarr: xr.DataArray, alpha: float) -> dict:
    """Get vmin, vmax using mean and std."""
    mean = xarr.mean()
    std = xarr.std()
    return {"vmin": max(0.0, mean - alpha * std), "vmax": mean + alpha * std}


def _set_real_aspect(axes: typing.Union[plt.Axes, typing.Iterable[plt.Axes]]) -> None:
    """Change all axes to be equal aspect."""
    if isinstance(axes, typing.Iterable):
        for ax in axes:
            _set_real_aspect(ax)
    else:
        axes.set_aspect(aspect="equal", adjustable="box")
    return


def _invert_yaxis(axes: typing.Union[plt.Axes, typing.Iterable[plt.Axes]]) -> None:
    """Change all axes to be equal aspect."""
    if isinstance(axes, typing.Iterable):
        for ax in axes:
            _invert_yaxis(ax)
    else:
        axes.invert_yaxis()
    return


def _draw_windows(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw windows on the axes."""
    n_c = len(COLORS)
    for i, row in enumerate(df.itertuples()):
        xy = (row.x - row.dx - 0.5, row.y - row.dy - 0.5)
        width = row.dx * 2 + 1
        height = row.dy * 2 + 1
        patch = patches.Rectangle(
            xy,
            width=width,
            height=height,
            linewidth=1,
            edgecolor=COLORS[i % n_c],
            fill=False,
        )
        ax.add_patch(patch)
    return


def _average_intensity(frame: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """Calculate the average intensity in windows. Return an array of average intensity."""
    if frame.ndim < 2:
        raise CrystalMapperError("frame must have no less than 2 dimensions.")
    elif frame.ndim > 2:
        n = frame.ndim - 2
        frame = frame.mean(axis=tuple(range(n)))
    ny, nx = frame.shape
    # create tasks
    I_in_windows = []
    for row in windows.itertuples():
        slice_y = slice(max(row.y - row.dy, 0), min(row.y + row.dy + 1, ny))
        slice_x = slice(max(row.x - row.dx, 0), min(row.x + row.dx + 1, nx))
        I_in_window = np.mean(frame[slice_y, slice_x], dtype=frame.dtype)
        I_in_windows.append(I_in_window)
    return np.array(I_in_windows)


def _track_peaks(frames: xr.DataArray, windows: pd.DataFrame) -> np.ndarray:
    """Create a list of tasks to compute the grain maps. Each task is one grain map."""
    # create intensity vs time for each grain
    global _VERBOSE
    intensities = []
    n = frames.shape[0]
    iter_range = range(n)
    if _VERBOSE > 0:
        iter_range = tqdm.tqdm(iter_range)
    for i in iter_range:
        intensity = _average_intensity(frames[i].values, windows)
        intensities.append(intensity)
    # axis: grain, frame
    return np.stack(intensities).transpose()


def _reshape_to_ndarray(arr: np.ndarray, metadata: dict) -> np.ndarray:
    if "shape" not in metadata:
        raise CrystalMapperError("Missing key '{}' in metadata.".format("shape"))
    shape = list(arr.shape)[:-1]
    shape.extend(metadata["shape"])
    arr: np.ndarray = arr.reshape(shape)
    # if snaking the row
    if (
        "snaking" in metadata
        and len(metadata["snaking"]) > 1
        and metadata["snaking"][1]
    ):
        if len(metadata["shape"]) != 2:
            raise CrystalMapperError("snaking only works for the 2 dimension array.")
        n = arr.shape[1]
        for i in range(n):
            if i % 2 == 1:
                arr[:, i, :] = arr[:, i, ::-1]
    return arr


def _create_windows_from_width(df: pd.DataFrame, width: int) -> pd.DataFrame:
    width = int(width)
    df2 = pd.DataFrame()
    df2["y"] = df["y"].apply(math.floor)
    df2["dy"] = width
    df2["x"] = df["x"].apply(math.floor)
    df2["dx"] = width
    return df2


def _min_and_max_along_time2(
    data: xr.DataArray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Extract the minimum and maximum values of each pixel in a series of mean frames. Return a data array.
    First is the min array and the second is the max array."""
    global _VERBOSE
    min_arr = max_arr = np.mean(data[0].values, axis=0)
    iter_range = range(1, len(data))
    if _VERBOSE > 0:
        iter_range = tqdm.tqdm(iter_range)
    for i in iter_range:
        arr = np.mean(data[i].values, axis=0)
        min_arr = np.fmin(min_arr, arr)
        max_arr = np.fmax(max_arr, arr)
    return min_arr, max_arr


def _get_coords(start_doc: dict) -> typing.List[np.ndarray]:
    """Get coordinates."""
    if "shape" not in start_doc:
        raise KeyError("Missing key '{}' in the metadata.".format("shape"))
    if "extents" not in start_doc:
        raise KeyError("Missing key '{}' in the metadata".format("extents"))
    shape = start_doc["shape"]
    extents = [np.asarray(extent) for extent in start_doc["extents"]]
    return [np.linspace(*extent, num) for extent, num in zip(extents, shape)]


def _limit_3std(da: xr.DataArray) -> typing.Tuple[float, float]:
    """Return the mean - 3 * std and mean + 3 * std of the data array.

    Parameters
    ----------
    da

    Returns
    -------

    """
    m, s = da.mean(), da.std()
    return m - 3 * s, m + 3 * s


def _plot_crystal_maps(
    da: xr.DataArray,
    limit_func: typing.Callable = None,
    invert_y: bool = False,
    **kwargs
) -> FacetGrid:
    """Plot the crystal maps.

    Parameters
    ----------
    da
    limit_func
    invert_y
    kwargs

    Returns
    -------

    """
    if limit_func is None:
        limit_func = _limit_3std
    kwargs.setdefault("col", "grain")
    kwargs.setdefault("col_wrap", 20)
    kwargs.setdefault("sharex", False)
    kwargs.setdefault("sharey", False)
    kwargs.setdefault("add_colorbar", False)
    vmin, vmax = limit_func(da)
    kwargs.setdefault("vmax", vmax)
    kwargs.setdefault("vmin", vmin)
    facet = da.plot.imshow(**kwargs)
    _set_real_aspect(facet.axes)
    if invert_y:
        if kwargs.get("sharey"):
            # if y is shared, only the first one need to be inverted
            _invert_yaxis(facet.axes.flatten()[0])
        else:
            _invert_yaxis(facet.axes)
    return facet


def _plot_rocking_curves(da: xr.DataArray, **kwargs) -> FacetGrid:
    """Plot the rocking curves.

    Parameters
    ----------
    da
    kwargs

    Returns
    -------

    """
    kwargs.setdefault("col", "grain")
    kwargs.setdefault("col_wrap", 5)
    kwargs.setdefault("sharex", False)
    kwargs.setdefault("sharey", False)
    return da.plot.line(**kwargs)


def _auto_plot(
    da: xr.DataArray,
    title: typing.Tuple[str, str] = None,
    invert_y: bool = False,
    **kwargs
) -> FacetGrid:
    """Automatically detect the data type and plot the data array.

    Parameters
    ----------
    da :
        The data array containing the intensity.
    title :
        Determine title of the axes. It is a format string and a name of the variable. If None, do nothing.
    invert_y :
        Invert the y axis.
    kwargs :
        The kwargs for the configuration of the plot.

    Returns
    -------
    Usually a FacetGrid object.
    """
    if da.ndim <= 1:
        facet = da.plot(**kwargs)
    elif da.ndim == 2:
        facet = _plot_rocking_curves(da, **kwargs)
    elif da.ndim == 3:
        facet = _plot_crystal_maps(da, invert_y=invert_y, **kwargs)
    else:
        kwargs.setdefault("col", "grain")
        facet = da.plot(**kwargs)
    if title is not None:
        v_name, f_title = title
        vals: np.ndarray = da[v_name].values
        axes: typing.List[plt.Axes] = facet.axes.flatten()
        for ax, val in zip(axes, vals):
            ax.set_title(f_title.format(val))
    facet.fig.tight_layout()
    return facet


def auto_plot_dataset(
    ds: xr.Dataset,
    key: str = "intensity",
    title: typing.Tuple[str, str] = None,
    invert_y: bool = False,
    **kwargs
):
    facet = _auto_plot(ds[key], title=None, invert_y=invert_y, **kwargs)
    if title is not None:
        v_name, f_title = title
        vals: np.ndarray = ds[v_name].values
        axes: typing.List[plt.Axes] = facet.axes.flatten()
        for ax, val in zip(axes, vals):
            ax.set_title(f_title.format(val))
    facet.fig.tight_layout()
    return facet


def _set_vlim(
    kwargs: dict,
    da: xr.DataArray,
    alpha: float,
    low_lim: float = 0.0,
    high_lim: float = float("inf"),
) -> None:
    mean = da.values.mean()
    std = da.values.std()
    kwargs.setdefault("vmin", max(low_lim, mean - std * alpha))
    kwargs.setdefault("vmax", min(high_lim, mean + std * alpha))
    return


def plot_windows(data: xr.Dataset, **kwargs) -> FacetGrid:
    # get difference
    diff = data["light"] - data["dark"]
    diff.attrs = data["light"].attrs
    # get dataframe
    df = data[["y", "dy", "x", "dx"]].to_dataframe()
    # plot
    facet = diff.plot.imshow(**kwargs)
    _draw_windows(df, facet.axes)
    # use real aspect
    _set_real_aspect(facet.axes)
    return facet


def _get_anlge(v1: np.ndarray, v2: np.ndarray) -> float:
    inner = np.dot(v1, v2)
    inner /= np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(inner))


class CrystalMapperError(Exception):
    pass


class CrystalMapper(object):
    """The Calculator of the crystal maps.

    Each of the attribute can be the calculated output and the input for the next step of calculation.

    Attributes
    ----------
    frames_arr : DataArray
        A data array of diffraction data. It is assumed to be a (N, F, Y, X) shape array, where N is the number
        of exposure, F is the number of the frames in one exposure, the Y is the number of pixels in vertical,
        X is the number of pixels in horizontal. It can also be (N, Y, X) shape array if there is only one
        frame in each exposure.
    dark : ndarray
        The minimum values on the pixels in whole series of exposures. It is assumed to be the background.
    light: ndarray
        The maximum values on the pixels in the whole series of exposures. It is assumed to be a image of all
        the Bragg peaks on the background.
    peaks: DataFrame
        A dataframe records all the peak positions.
    windows: DataFrame
        A dataframe records all the center positions and half width of the windows.
    intensity: ndarray
        The array of the average intensity in each of the windows in the series of exposures. It has the shape (
        W, dim_0, dim_1, ...). The W is the number of the windows and the dim_i is the length of the dimension
        i in the grid scan.
    bkg_intensity : ndarray
        The integrated background intensity in the windows.
    coords : ndarray
        The coordinates of the dim_0, dim_1, ... in grid scan.
    metadata : dict
        The dictionary to record the metadata. There are two important keys: shape, extents. The shape is the
        number of points in each dimension in the grid scan. It should be a array like (dim_0, dim_1, ...). The
        extents are the start and end value for each dimension. It should be array like [(start0, end0),
        (start1, end1), ...]. These two keys are used to calculate the coordinates.
    ai : AzimuthalIntegrator
        The class to hold the data of the geometry of the experiments. It is used to index the Q values for each
        window.
    """

    def __init__(self):
        # dims: time, ..., pixel_y, pixel_x
        self.frames_arr: typing.Union[None, xr.DataArray] = None
        # dims: pixel_y, pixel_x
        self.dark: typing.Union[None, np.ndarray] = None
        # dims: pixel_y, pixel_x
        self.light: typing.Union[None, np.ndarray] = None
        # index: all_grain
        self.peaks: typing.Union[None, pd.DataFrame] = None
        # index: grain
        self.windows: typing.Union[None, pd.DataFrame] = None
        # dims: grain, dim_1, ..., dim_n
        self.intensity: typing.Union[None, np.ndarray] = None
        # dims: grain
        self.bkg_intensity: typing.Union[None, np.ndarray] = None
        # dims: dim_1, ..., dim_n
        self.coords: typing.Union[None, typing.List[np.ndarray]] = None
        # keys: shape, extents, snaking
        self.metadata: typing.Union[None, dict] = None
        # pyFAI
        self.ai: typing.Union[None, AzimuthalIntegrator] = None
        # dims: all two theta
        self.all_twotheta: typing.Union[None, np.ndarray] = None
        # dims: all d spacings
        self.all_dspacing: typing.Union[None, np.ndarray] = None
        # dims: all d spacings, hkl, element in hkl
        self.all_hkl: typing.Union[None, np.ndarray] = None
        # dims: all d spacings
        self.all_n_hkl: typing.Union[None, np.ndarray] = None
        # dims: grain, d_idx
        self.dspacing: typing.Union[None, np.ndarray] = None
        # dims: grain, d_idx, hkl_idx ,hkl
        self.hkl: typing.Union[None, np.ndarray] = None
        # dims: grain, d_idx
        self.n_hkl: typing.Union[None, np.ndarray] = None
        # a pyFAI cell object
        self.cell: typing[None, Cell] = None
        # column names of the windows
        self.window_names = frozenset(["x", "y", "dx", "dy", "d", "Q"])
        # UBmatrix object
        self.ubmatrix: UBMatrix = UBMatrix()
        # Result of peak indexing
        self.peak_indexes = None

    def _check_attr(self, name: str):
        if getattr(self, name) is None:
            raise CrystalMapperError(
                "Attribute '{}' is None. Please set it.".format(name)
            )
        if name == "metadata" and "shape" not in self.metadata:
            raise CrystalMapperError("There is no key 'shape' in the metadata.")

    def _squeeze_shape_and_extents(self) -> None:
        """Squeeze the shape and extents so that it only has the dimension with length > 1."""
        self._check_attr("metadata")
        shape = self.metadata["shape"]
        n = len(shape)
        # get the index of the dimension > 1
        index = {i for i in range(n) if shape[i] > 1}
        self.metadata["shape"] = [shape[i] for i in range(n) if i in index]
        if "extents" in self.metadata:
            extents = self.metadata["extents"]
            self.metadata["extents"] = [extents[i] for i in range(n) if i in index]
        return

    def calc_dark_and_light_from_frames_arr(self, index_range: slice = None):
        """Get the light and dark frame in a series of frames."""
        self._check_attr("frames_arr")
        frames_arr = (
            self.frames_arr[index_range] if index_range is not None else self.frames_arr
        )
        self.dark, self.light = _min_and_max_along_time2(frames_arr)
        return

    def calc_peaks_from_dk_sub_frame(
        self, radius: typing.Union[int, tuple], *args, **kwargs
    ):
        """Get the Bragg peaks on the light frame."""
        self._check_attr("light")
        light = self.light if self.dark is None else self.light - self.dark
        self.peaks = tp.locate(light, 2 * radius + 1, *args, **kwargs)
        return

    def calc_windows_from_peaks(self, max_num: int, width: int):
        """Gte the windows for the most brightest Bragg peaks."""
        self._check_attr("peaks")
        if self.peaks.shape[0] == 0:
            raise CrystalMapperError(
                "There is no peak found on the image. Please check your peaks table."
            )
        df = self.peaks.nlargest(max_num, "mass")
        self.windows = _create_windows_from_width(df, width)
        return

    def calc_intensity_in_windows(self):
        """Get the intensity array as a function of index of frames."""
        self._check_attr("frames_arr")
        self._check_attr("windows")
        self.intensity = _track_peaks(self.frames_arr, self.windows)
        if self.dark is not None:
            self.bkg_intensity = _average_intensity(self.dark, self.windows)
            self.intensity = (
                self.intensity.transpose() - self.bkg_intensity
            ).transpose()
        else:
            _my_print("Attribute 'dark' is None. No background correction.")
        return

    def assign_q_values(self) -> None:
        """Assign the values to the windows dataframe."""
        self._check_attr("ai")
        self._check_attr("windows")
        # change the unit of Q to inverse A
        qa = self.ai.qArray() / 10.0
        self.windows["Q"] = [qa[row.y, row.x] for row in self.windows.itertuples()]
        return

    def assign_d_values(self) -> None:
        self.windows["d"] = 2.0 * math.pi / self.windows["Q"]
        return

    def _Q_to_twotheta(self, Q: np.ndarray):
        self._check_attr("ai")
        # change wavelength unit to A
        w = self.ai.wavelength * 1e10
        return np.rad2deg(2.0 * np.arcsin(Q * w / (4.0 * math.pi)))

    def _d_to_twotheta(self, d: np.ndarray):
        return self._Q_to_twotheta(2.0 * math.pi / d)

    def assign_twotheta_values(self) -> None:
        # wavelength meter, Q inverse nanometer
        self.windows["twotheta"] = self._Q_to_twotheta(self.windows["Q"])
        return

    def reshape_intensity(self) -> None:
        """Reshape the intensity array."""
        self._check_attr("metadata")
        self._check_attr("intensity")
        self.intensity = _reshape_to_ndarray(self.intensity, self.metadata)
        return

    def calc_coords(self) -> None:
        """Calculate the coordinates."""
        self._check_attr("metadata")
        self.coords = _get_coords(self.metadata)
        return

    def dark_to_xarray(self) -> xr.DataArray:
        """Convert the dark image to DataArray."""
        self._check_attr("dark")
        return xr.DataArray(self.dark, dims=["pixel_y", "pixel_x"])

    def light_to_xarray(self) -> xr.DataArray:
        """Convert the light image to DataArray"""
        self._check_attr("light")
        return xr.DataArray(self.light, dims=["pixel_y", "pixel_x"])

    def intensity_to_xarray(self) -> xr.DataArray:
        """Convert the intensity array to DataArray"""
        self._check_attr("intensity")
        dims = ["dim_{}".format(i) for i in range(self.intensity.ndim - 1)]
        arr = xr.DataArray(self.intensity, dims=["grain"] + dims)
        if self.coords is not None:
            coords = self.coords_to_dict()
            arr = arr.assign_coords(coords)
        return arr

    def windows_to_xarray(self) -> xr.DataArray:
        """Convert the windows DataFrame to xarray."""
        self._check_attr("windows")
        return self.windows.rename_axis(index="grain").to_xarray()

    def coords_to_dict(self) -> dict:
        """Convert the coordinates to dictionary."""
        self._check_attr("coords")
        return {"dim_{}".format(i): coord for i, coord in enumerate(self.coords)}

    def dspacing_to_xarray(self) -> xr.DataArray:
        return xr.DataArray(
            self.dspacing, dims=["grain", "d_idx"], attrs={"units": r"nm$^{-1}$"}
        )

    def hkl_to_xarray(self) -> xr.DataArray:
        return xr.DataArray(self.hkl, dims=["grain", "d_idx", "hkl_idx", "reciprocal"])

    def n_hkl_to_xarray(self) -> xr.DataArray:
        return xr.DataArray(self.n_hkl, dims=["grain", "d_idx"])

    def to_dataset(self) -> xr.Dataset:
        """Convert the calculation results to DataSet."""
        dct = dict()
        if self.dark is not None:
            dct["dark"] = self.dark_to_xarray()
        if self.light is not None:
            dct["light"] = self.light_to_xarray()
        if self.intensity is not None:
            dct["intensity"] = self.intensity_to_xarray()
        ds = xr.Dataset(dct)
        if self.windows is not None:
            ds2 = self.windows_to_xarray()
            ds = ds.merge(ds2)
        if self.dspacing is not None:
            ds = ds.assign({"dspacing": self.dspacing_to_xarray()})
        if self.hkl is not None:
            ds = ds.assign({"hkl": self.hkl_to_xarray()})
        if self.n_hkl is not None:
            ds = ds.assign({"n_hkl": self.n_hkl_to_xarray()})

        if self.metadata is not None:
            ds = ds.assign_attrs(**self.metadata)
        return ds

    def get_frame(self, index: int) -> xr.DataArray:
        """Get the frame of at the index."""
        self._check_attr("frames_arr")
        frames = self.frames_arr[index].compute()
        if frames.ndim == 3:
            return frames.mean(axis=0)
        elif frames.ndim == 2:
            return frames
        else:
            raise CrystalMapperError(
                "The dimension of the frame is {}. Require 2 or 3.".format(frames.ndim)
            )

    def show_frame(self, index: int, *args, **kwargs) -> FacetGrid:
        """Show the frame at that index."""
        self._check_attr("frames_arr")
        frame = self.get_frame(index)
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_windows_on_frame(self, index: int, *args, **kwargs) -> FacetGrid:
        """Show the windows on the frame at the index."""
        self._check_attr("windows")
        facet = self.show_frame(index, *args, **kwargs)
        _draw_windows(self.windows, facet.axes)
        return facet

    def show_dark(self, *args, **kwargs) -> FacetGrid:
        """Show the dark image."""
        self._check_attr("dark")
        frame = self.dark_to_xarray()
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_light(self, *args, **kwargs) -> FacetGrid:
        """Show the light image."""
        self._check_attr("light")
        frame = self.light_to_xarray()
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_light_sub_dark(self, *args, **kwargs) -> FacetGrid:
        """Show the dark subtracted light image."""
        self._check_attr("light")
        light = self.light_to_xarray()
        if self.dark is not None:
            dark = self.dark_to_xarray()
            light = np.subtract(light, dark)
        _set_vlim(kwargs, light, 4.0)
        facet = light.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_windows(self, *args, **kwargs) -> FacetGrid:
        """Show the windows on the dark subtracted light image."""
        self._check_attr("light")
        self._check_attr("windows")
        facet = self.show_light_sub_dark(*args, **kwargs)
        _draw_windows(self.windows, facet.axes)
        return facet

    def show_intensity(self, **kwargs) -> FacetGrid:
        """Show the intensity array."""
        arr = self.intensity_to_xarray()
        return _auto_plot(arr, **kwargs)

    @staticmethod
    def auto_visualize(
        ds: xr.Dataset,
        key: str = "intensity",
        title: typing.Tuple[str, str] = None,
        invert_y: bool = False,
        **kwargs
    ) -> FacetGrid:
        """Automatically plot the intensity array in the dataset."""
        return auto_plot_dataset(ds, key, title, invert_y, **kwargs)

    def auto_process(
        self,
        num_wins: int = 100,
        wins_width: int = 25,
        kernel_radius: int = 25,
        index_filter: slice = None,
        dspacing_tolerance: typing.Tuple[float, float] = None,
        **kwargs
    ) -> None:
        """Automatically process the data in the standard protocol.

        The calculation results are saved in attributes.

        Parameters
        ----------
        num_wins : int
            The number of windows.
        wins_width : int
            The half width of the windows in pixels.
        kernel_radius : int
            The radius of the kernel to use in peak finding in pixels. It must be an odd integer.
        index_filter : slice
            The index slice of the data to use in the calculation of the dark and light image..
        dspacing_tolerance : tuple
            The tolerance to find the d-spacing for ech peak. It is the ratio between the expected and real.
        kwargs :
            The keyword arguments of the peak finding function `trackpy.locate`.

        Returns
        -------
        None.
        """
        self._squeeze_shape_and_extents()
        self.calc_dark_and_light_from_frames_arr(index_filter)
        self.calc_peaks_from_dk_sub_frame(kernel_radius, **kwargs)
        self.calc_windows_from_peaks(num_wins, wins_width)
        self.calc_intensity_in_windows()
        try:
            self.calc_coords()
        except CrystalMapperError as e:
            print(e)
        try:
            self.assign_q_values()
            self.assign_d_values()
        except CrystalMapperError as e:
            print(e)
        if dspacing_tolerance is not None:
            try:
                self.calc_hkls_in_a_range(*dspacing_tolerance)
            except CrystalMapperError as e:
                print(e)
        try:
            self.reshape_intensity()
        except CrystalMapperError as e:
            print(e)
        return

    def load_ai(self, filename: str) -> None:
        ai = pyFAI.load(filename)
        self.ai = ai
        self.ubmatrix.geo = ai
        return

    def load_frames_arr(self, filename: str) -> None:
        self.frames_arr = xr.load_dataarray(filename)
        return

    def load_dataset(self, ds: xr.Dataset) -> None:
        w_names = []
        for name in ds:
            if name not in self.window_names:
                self.__setattr__(str(name), ds[name].values)
            else:
                w_names.append(name)
        self.windows = ds[w_names].to_dataframe()
        return

    def load_cell(self, filename: str) -> None:
        with pathlib.Path(filename).open("r") as f:
            dct = yaml.safe_load(f)
        self.cell = Cell(**dct)
        return

    def load_cell_from_cif(self, cif_file: str) -> None:
        stru: Structure = loadStructure(cif_file, fmt="cif")
        lat: Lattice = stru.lattice
        self.cell = Cell(
            a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma
        )
        return

    def calc_hkls_in_a_range(self, lb: float, rb: float):
        """Calculate the hkl in a range of Q value."""
        self._check_attr("windows")
        self._check_attr("cell")
        if "d" not in self.windows.columns:
            if "Q" not in self.windows.columns:
                self.assign_q_values()
            self.assign_d_values()
        if self.all_dspacing is None:
            self._calc_ds_and_hkls()
        dspacings = []
        hkls = []
        n_hkls = []
        for d in self.windows["d"]:
            l_idx, r_idx = self._search_hkls_idx(d, lb, rb)
            dspacing = self.all_dspacing[l_idx:r_idx]
            dspacings.append(dspacing)
            hkl = self.all_hkl[l_idx:r_idx]
            hkls.append(hkl)
            n_hkl = self.all_n_hkl[l_idx:r_idx]
            n_hkls.append(n_hkl)
        self.dspacing = self._stack_arrays(dspacings)
        self.hkl = self._stack_arrays(hkls)
        self.n_hkl = self._stack_arrays(n_hkls)
        return

    def _calc_ds_and_hkls(self) -> None:
        dmin = self.windows["d"].min()
        dhkls = sorted(self.cell.d_spacing(dmin).values())
        if len(dhkls) == 0:
            raise CrystalMapperError(
                "There is no matching d-spacing. Please check the cell attribute."
            )
        ds = []
        hkls = []
        rows = []
        for dhkl in dhkls:
            d = dhkl[0]
            ds.append(d)
            hkl = np.asarray(dhkl[1:])
            rows.append(hkl.shape[0])
            hkls.append(hkl)
        max_row = max(rows)
        for i, hkl in enumerate(hkls):
            # pad the hkls to be the same number of rows
            hkls[i] = self._pad_array(hkl, (max_row, 3))
        # reverse the order to make sure the two theta is increasing
        self.all_dspacing = np.asarray(ds[::-1])
        self.all_twotheta = self._d_to_twotheta(self.all_dspacing)
        self.all_hkl = np.asarray(hkls[::-1])
        self.all_n_hkl = np.asarray(rows[::-1])
        return

    def _search_hkls_idx(
        self, d: float, lb: float, rb: float
    ) -> typing.Tuple[int, int]:
        ratio = np.divide(self.all_dspacing, d)
        return bisect.bisect(ratio, lb), bisect.bisect(ratio, rb)

    def calc_hkls_lower_and_upper(self, two_theta_tolerance: float) -> None:
        """Calculate hkls and assign them to the peaks.

        Find the upper and lower bound of the Q for each Q value in the dataframe. The hkls that have the upper
        and lower bound values are the possible hkls for that peak. The index of the Q value is the index of the
        group of possible hkls. The index is recorded in the dataframe for both upper and lower bound.
        """
        dtt = two_theta_tolerance
        del two_theta_tolerance
        self._check_attr("windows")
        self._check_attr("cell")
        if "twotheta" not in self.windows.columns:
            if "Q" not in self.windows.columns:
                self.assign_q_values()
            self.assign_d_values()
            self.assign_twotheta_values()
        if self.all_dspacing is None:
            self._calc_ds_and_hkls()
        uppers, lowers = [], []
        for tt in self.windows["twotheta"]:
            l, u = self._search_bounds(tt - dtt, tt + dtt)
            uppers.append(u)
            lowers.append(l)
        self.windows["lower_idx"] = lowers
        self.windows["upper_idx"] = uppers
        return

    def calc_diff_2theta(self) -> None:
        idxs = []
        diffs = []
        for tt, lower, upper in zip(
            self.windows["twotheta"],
            self.windows["lower_idx"],
            self.windows["upper_idx"],
        ):
            ltt = self.all_twotheta[lower] if lower >= 0 else float("-inf")
            utt = self.all_twotheta[upper] if upper >= 0 else float("inf")
            l_diff, u_diff = tt - ltt, utt - tt
            idx, diff = (lower, l_diff) if l_diff <= u_diff else (upper, u_diff)
            idxs.append(idx)
            diffs.append(diff)
        self.windows["closet_idx"] = idxs
        self.windows["diff"] = diffs
        return

    def calc_hkls(self, two_theta_tolerance: float) -> None:
        """Find the losest d spacing for each peak and record its index. Use the dspacing to find possible hkls.

        Returns
        -------
        None. The results are saved in self.windows.
        """
        self.calc_hkls_lower_and_upper(two_theta_tolerance)
        return

    def _search_bounds(
        self, lower_tt: float, upper_tt: float
    ) -> typing.Tuple[int, int]:
        left = np.searchsorted(self.all_twotheta, lower_tt, side="left")
        right = max(np.searchsorted(self.all_twotheta, upper_tt, side="right") - 1, 0)
        # if no peaks found, use the closet
        if right < left:
            d1, d2 = (
                lower_tt - self.all_twotheta[right],
                upper_tt - self.all_twotheta[left],
            )
            if d1 < d2:
                left = right
            else:
                right = left
        return left, right

    def load_structure(self, cif_file: str):
        """Load the structure of the sample. The `cell` and `ubmatrix.lattice` will be loaded."""
        self.ubmatrix.set_lat_from_cif(cif_file)
        lat = self.ubmatrix.lat
        self.cell = Cell(
            a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma
        )
        return

    def search_two_peaks(self, peaks: typing.List[int]) -> typing.Tuple[int, int]:
        """Find the two peaks that have the smallest difference between the measured dspacings and those in
        structure. Return their index.

        Parameters
        ----------
        peaks

        Returns
        -------

        """
        self._check_attr("windows")
        # check
        if (
            "diff" not in self.windows.columns
            or "closet_idx" not in self.windows.columns
        ):
            raise CrystalMapperError("Please run calc_hkls first.")
        # get
        df = self.windows.loc[peaks]
        # run
        sel_df: pd.DataFrame = df.nsmallest(2, columns=["diff_dspacing"])
        if sel_df.shape[0] < 2:
            raise CrystalMapperError(
                "There are less than 2 peaks found. Please check the `windows` dataframe."
            )
        return tuple(sel_df.index.tolist())

    def _get_hkls(self, left: int, right: int):
        ns = self.all_n_hkl[left:right + 1]
        hklss = self.all_hkl[left:right + 1]
        return np.concatenate([hkls[:n] for n, hkls in zip(ns, hklss)])

    def _loss(self, hkls: np.ndarray) -> float:
        cost = self._get_losses(hkls)
        return np.min(cost)

    def _get_losses(self, hkls):
        rhkls = np.around(hkls)
        vs = self.ubmatrix.reci_to_cart(hkls)
        rvs = self.ubmatrix.reci_to_cart(rhkls)
        diffs_sq = np.sum((rvs - vs) ** 2, axis=1)
        lens_sq = np.sum(vs**2, axis=1)
        cost = np.sqrt(diffs_sq / lens_sq)
        return cost

    def index_peaks(
        self, peak1: int, peak2: int, others: typing.List[int]
    ) -> typing.Generator[xr.Dataset, None, None]:
        """Use the hkls of two peaks to calculate U matrixs and use them to index other peaks. Return the hkls.

        Parameters
        ----------
        peak1
        peak2
        others

        Returns
        -------

        """
        self._check_attr("all_n_hkl")
        self._check_attr("all_hkl")
        self._check_attr("windows")
        self._set_us_for_peaks(peak1, peak2)
        hkls1 = self._get_hkls_for_peak(peak1)
        hkls2 = self._get_hkls_for_peak(peak2)
        # index the hkls
        peaks = np.array([peak1, peak2] + others)
        n1, n2 = hkls1.shape[0], hkls2.shape[0]
        for i in range(n1):
            for j in range(n2):
                ds = self._index_others(hkls1[i], hkls2[j], others)
                ds = ds.assign_coords({"peak": peaks})
                yield ds
        return

    def _get_hkls_for_peak(self, peak1: int) -> np.ndarray:
        row1 = self.windows.loc[peak1]
        l1, r1 = int(row1["lower_idx"]), int(row1["upper_idx"])
        # a list of hkls, zero padded
        hkls1 = self._get_hkls(l1, r1)
        return hkls1

    def _set_us_for_peaks(self, peak1: int, peak2: int) -> None:
        row1 = self.windows.loc[peak1]
        row2 = self.windows.loc[peak2]
        xy1 = np.array([row1["x"], row1["y"]])
        xy2 = np.array([row2["x"], row2["y"]])
        self.ubmatrix.set_u1_from_xy(xy1)
        self.ubmatrix.set_u2_from_xy(xy2)
        return

    def _index_others(
        self, h1: np.ndarray, h2: np.ndarray, others: typing.List[int]
    ) -> xr.Dataset:
        self.ubmatrix.set_h1_from_hkl(h1)
        self.ubmatrix.set_h2_from_hkl(h2)
        self.ubmatrix.get_U()
        res = [h1, h2]
        for k in others:
            row = self.windows.loc[k]
            xy = np.array([row["x"], row["y"]])
            u = self.ubmatrix.xy_to_lab(xy)
            v = self.ubmatrix.lab_to_cart(u)
            hkl = self.ubmatrix.cart_to_reci(v)
            res.append(hkl)
        res = np.asarray(res)
        lval = self._loss(res[2:])
        ds = xr.Dataset({"hkls": (["peak", "hkl"], res), "loss": lval})
        return ds

    def index_peaks_in_one_grain(
        self, peaks: typing.List[int]
    ) -> typing.Generator[xr.Dataset, None, None]:
        """Find the two peaks that have the smallest difference between the measured dspacings and those in
        structure. Use the hkls of two peaks to calculate U matrixs and use them to index other peaks.
        Return the hkls.

        Parameters
        ----------
        peaks

        Returns
        -------

        """
        for p1, p2 in itertools.combinations(peaks, 2):
            others = [p for p in peaks if p != p1 and p != p2]
            yield from self.index_peaks(p1, p2, others)
        return

    @staticmethod
    def _pad_array(arr: np.ndarray, shape: typing.Sequence[int]) -> np.ndarray:
        lst = [(0, s1 - s2) for s1, s2 in zip(shape, arr.shape)]
        return np.pad(arr, lst, constant_values=0)

    @staticmethod
    def _get_max_shape(arrs: typing.Sequence[np.ndarray]):
        max_shape = np.shape(arrs[0])
        n = len(arrs)
        for i in range(1, n):
            max_shape = np.fmax(max_shape, np.shape(arrs[i]))
        return max_shape

    def _stack_arrays(self, arrs: typing.Sequence[np.ndarray]) -> np.ndarray:
        max_shape = self._get_max_shape(arrs)
        arrs = [self._pad_array(a, max_shape) for a in arrs]
        return np.stack(arrs)

    def _get_angle_in_sample_frame(self) -> float:
        return _get_anlge(self.ubmatrix.u1, self.ubmatrix.u2)

    def _get_anlge_in_grain_frame(self) -> float:
        return _get_anlge(self.ubmatrix.h1, self.ubmatrix.h2)

    def _get_anlge_h1_h2(
        self, peak1: int, peak2: int
    ) -> typing.Generator[tuple, None, None]:
        hkls1 = self._get_hkls_for_peak(peak1)
        hkls2 = self._get_hkls_for_peak(peak2)
        self._set_us_for_peaks(peak1, peak2)
        angle0 = self._get_angle_in_sample_frame()
        n1 = hkls1.shape[0]
        n2 = hkls2.shape[0]
        for i in range(n1):
            for j in range(n2):
                self.ubmatrix.set_h1_from_hkl(hkls1[i])
                self.ubmatrix.set_h2_from_hkl(hkls2[j])
                angle = self._get_anlge_in_grain_frame()
                if 1e-8 < abs(angle) < (180.0 - 1e-8):
                    yield abs(
                        angle - angle0
                    ), angle0, angle, self.ubmatrix.h1, self.ubmatrix.h2
        return

    def _get_hkl_for_a_peak(self, peak: int) -> HKL:
        row = self.windows.loc[peak]
        xy = np.array([row["x"], row["y"]])
        u = self.ubmatrix.xy_to_lab(xy)
        v = self.ubmatrix.lab_to_cart(u)
        hkl = self.ubmatrix.cart_to_reci(v)
        return hkl

    def _get_indexing_result_for_peaks(self, peaks: typing.List[int]) -> np.ndarray:
        return np.stack([self._get_hkl_for_a_peak(peak) for peak in peaks])

    def _index_peaks2(
        self, peak1: int, peak2: int, peaks: typing.List[int]
    ) -> typing.Generator[xr.Dataset, None, None]:
        lst = sorted(self._get_anlge_h1_h2(peak1, peak2), key=(lambda tup: tup[0]))
        for da, a0, a, h1, h2 in lst:
            self.ubmatrix.h1 = h1
            self.ubmatrix.h2 = h2
            self.ubmatrix.get_U()
            hkls = self._get_indexing_result_for_peaks(peaks)
            loss = self._loss(hkls)
            yield xr.Dataset(
                {
                    "hkls": (["peak", "hkl"], hkls),
                    "loss": loss,
                    "angle_sample": a0,
                    "angle_grain": a,
                    "diff_angle": da,
                    "peak1": peak1,
                    "peak2": peak2,
                }
            )
        return

    def index_peaks_in_one_grain2(
        self, peaks: typing.List[int], first_n: int, index_all: bool = False
    ) -> xr.Dataset:
        """Guess the index of the peaks in one grain.

        Parameters
        ----------
        peaks : typing.List[int]
            The index of the peaks in the table.
        first_n : int
            The number of the best guess to record for each combination of peaks.
        index_all : bool, optional
            If True, index all the peaks dataset. Else, index only the input peaks from one grain. by default False

        Returns
        -------
        A xarray dataset contain the guess. It contains candidate, peaks and hkls dimensions.
        The candidate is one possible guess. The peaks are rows of all hkls. And the hkls are the
        h, k and l index in that row.
        """
        n = len(peaks)
        all_peaks = self.windows.index.values if index_all else peaks

        def _gen():
            for i, j in itertools.permutations(range(n), 2):
                for m, result in enumerate(
                    self._index_peaks2(peaks[i], peaks[j], all_peaks)
                ):
                    if m > first_n - 1:
                        break
                    yield result.expand_dims("candidate")

        final = xr.concat(_gen(), dim="candidate")
        final = final.assign_coords({"peak": all_peaks})
        self.peak_indexes = final
        return final

    def _print_group(self, data: xr.Dataset) -> None:
        print(
            "Use peak {} and peak {} in the indexing.".format(
                data["peak1"].item(), data["peak2"].item()
            )
        )
        header = self._get_header_df(data)
        print(_tableize(header))
        print("Below is the prediction.")
        body = self._get_body_df(data)
        print(_tableize(body))
        print()
        return

    def _get_body_df(self, data):
        dct = defaultdict(list)
        n = data.sizes["peak"]
        for i in range(n):
            sel = data.isel({"peak": i})
            dct["peak"].append(sel["peak"].item())
            dct["predicted hkl"].append(
                "{:.2f}, {:.2f}, {:.2f}".format(*sel["hkls"].values)
            )
            dct["rounded predicted hkl"].append(
                "{:.0f}, {:.0f}, {:.0f}".format(*sel["hkls"].values)
            )
        df = pd.DataFrame(dct)
        return df

    def _get_header_df(self, data) -> pd.DataFrame:
        dct = defaultdict(list)
        dct["angle in sample frame"].append(
            "{:.2f}".format(data["angle_sample"].item())
        )
        dct["angle in grain frame"].append("{:.2f}".format(data["angle_grain"].item()))
        dct["difference in angles"].append("{:.2f}".format(data["diff_angle"].item()))
        dct["badness of the prediction"].append("{:.4f}".format(data["loss"].item()))
        df = pd.DataFrame(dct)
        return df

    def print_indexing_result(self, data: xr.Dataset) -> None:
        """Print out the indexing results.

        Parameters
        ----------
        data : xr.Dataset
            The data set of returned by the index_peaks_in_one_grain2.
        """
        data = data.sortby("loss")
        n = data.sizes["candidate"]
        for i in range(n):
            sel = data.isel({"candidate": i})
            self._print_group(sel)
        return

    def show_peaks(
        self, data: xr.Dataset, peaks: typing.List[int], size: float = 5.0
    ) -> None:
        """Show the crystal maps of certain peaks.

        Parameters
        ----------
        peaks : typing.List[int]
            A list of integer of the peaks.
        size: float
            The size of one cystal map.
        """
        sel = data.sel({"grain": peaks})
        shape = data["intensity"].shape
        facet = auto_plot_dataset(
            sel, invert_y=True, col_wrap=10, aspect=shape[2] / shape[1], size=size
        )
        facet.fig.tight_layout()
        plt.show()
        return
