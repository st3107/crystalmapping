import bisect
import itertools
import math
import pathlib
import sys
import typing
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyFAI
import tqdm
import trackpy as tp
import xarray as xr
import yaml
from diffpy.structure import Lattice, Structure, loadStructure
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import Cell
from xarray.plot import FacetGrid

from .ubmatrix import UBMatrix

_VERBOSE = 1
COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
HKL = np.ndarray
Matrix = np.ndarray
T = typing
BlueskyRun = typing.Any


def _str_matrix(data: np.ndarray) -> str:
    s = [[str(e) for e in row] for row in data]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = "\t".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    return "\n".join(table)


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


def _get_n_largest(lst: T.Iterable[T.Any], n: int) -> T.List[T.Tuple]:
    res = []
    for item in lst:
        heappush(res, item)
        if len(res) > n:
            heappop(res)
    return res


def set_verbose(level: int) -> None:
    global _VERBOSE
    _VERBOSE = level


def _my_print(*args, **kwargs) -> None:
    if _VERBOSE > 0:
        print(*args, **kwargs)


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
    return max(0.0, m - 3 * s), m + 3 * s


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
    kwargs.setdefault("size", 5.0)
    kwargs.setdefault(
        "aspect",
    )
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
    da: xr.DataArray, title: typing.Tuple[str, str], invert_y: bool, **kwargs
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
    invert_y: bool = True,
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


def _get_anlge(v1: np.ndarray, v2: np.ndarray) -> float:
    inner = np.dot(v1, v2)
    inner /= np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(inner))


def _show_crystal_maps(
    data: xr.Dataset, peaks: T.Optional[typing.List[int]], **kwargs
) -> None:
    if peaks is not None:
        data = data.sel({"grain": peaks})
    facet = auto_plot_dataset(data, **kwargs)
    facet.fig.tight_layout()
    return


@dataclass
class AngleComparsion(object):

    h1: HKL
    h2: HKL
    angle_sample: float
    angle_grain: float
    diff_angle: float

    def __eq__(self, __o: object) -> bool:
        return self.diff_angle == __o.diff_angle

    def __lt__(self, __o: object) -> bool:
        return self.diff_angle > __o.diff_angle


@dataclass
class CrystalMapperConfig(object):

    image_data_key: str = "dexela_image"
    RoI_number: int = 100
    RoI_half_width: int = 25
    trackpy_kernel_size: int = 25
    slice_of_frames: T.Optional[slice] = None
    dspacing_bounds: T.Optional[T.Tuple[float, float]] = None
    index_tth_tolerance: float = 0.1
    index_best_n: int = 3
    index_all_peaks: bool = True


class CrystalMapperError(Exception):
    pass


@dataclass
class CrystalMapper(object):
    """The Calculator of the crystal maps.
    """

    # configuration
    _config: T.Optional[CrystalMapperConfig] = None
    # dims: time, ..., pixel_y, pixel_x
    _frames_arr: T.Optional[xr.DataArray] = None
    # dims: pixel_y, pixel_x
    _dark: T.Optional[np.ndarray] = None
    # dims: pixel_y, pixel_x
    _light: T.Optional[np.ndarray] = None
    # index: all_grain
    _peaks: T.Optional[pd.DataFrame] = None
    # index: grain
    _windows: T.Optional[pd.DataFrame] = None
    # dims: grain, dim_1, ..., dim_n
    _intensity: T.Optional[np.ndarray] = None
    # dims: grain
    _bkg_intensity: T.Optional[np.ndarray] = None
    # dims: dim_1, ..., dim_n
    _coords: T.Optional[typing.List[np.ndarray]] = None
    # keys: shape, extents, snaking
    _metadata: T.Optional[dict] = None
    # pyFAI
    _ai: T.Optional[AzimuthalIntegrator] = None
    # dims: all two theta
    _all_twotheta: T.Optional[np.ndarray] = None
    # dims: all d spacings
    _all_dspacing: T.Optional[np.ndarray] = None
    # dims: all d spacings, hkl, element in hkl
    _all_hkl: T.Optional[np.ndarray] = None
    # dims: all d spacings
    _all_n_hkl: T.Optional[np.ndarray] = None
    # dims: grain, d_idx
    _dspacing: T.Optional[np.ndarray] = None
    # dims: grain, d_idx, hkl_idx ,hkl
    _hkl: T.Optional[np.ndarray] = None
    # dims: grain, d_idx
    _n_hkl: T.Optional[np.ndarray] = None
    # a pyFAI cell object
    _cell: T.Optional[Cell] = None
    # column names of the windows
    _window_names = frozenset(["x", "y", "dx", "dy", "d", "Q"])
    # UBmatrix object
    _ubmatrix: UBMatrix = UBMatrix()
    # structure
    _stru: T.Optional[Structure] = None
    # crystal mapping results
    _crystal_maps: T.Optional[xr.Dataset] = None
    # peak indexing results
    _peak_index: T.Optional[xr.Dataset] = None

    def _check_attr(self, name: str):
        if getattr(self, "_{}".format(name)) is None:
            raise CrystalMapperError(
                "Attribute '{}' is None. Please set it.".format(name)
            )
        if name == "metadata" and "shape" not in self._metadata:
            raise CrystalMapperError("There is no key 'shape' in the metadata.")

    def _squeeze_shape_and_extents(self) -> None:
        """Squeeze the shape and extents so that it only has the dimension with length > 1."""
        self._check_attr("metadata")
        shape = self._metadata["shape"]
        n = len(shape)
        # get the index of the dimension > 1
        index = {i for i in range(n) if shape[i] > 1}
        self._metadata["shape"] = [shape[i] for i in range(n) if i in index]
        if "extents" in self._metadata:
            extents = self._metadata["extents"]
            self._metadata["extents"] = [extents[i] for i in range(n) if i in index]
        return

    def _calc_dark_and_light_from_frames_arr(self, index_range: slice = None):
        """Get the light and dark frame in a series of frames."""
        self._check_attr("frames_arr")
        frames_arr = (
            self._frames_arr[index_range]
            if index_range is not None
            else self._frames_arr
        )
        self._dark, self._light = _min_and_max_along_time2(frames_arr)
        return

    def _calc_peaks_from_dk_sub_frame(
        self, radius: typing.Union[int, tuple], *args, **kwargs
    ):
        """Get the Bragg peaks on the light frame."""
        self._check_attr("light")
        light = self._light if self._dark is None else self._light - self._dark
        self._peaks = tp.locate(light, 2 * radius + 1, *args, **kwargs)
        return

    def _calc_windows_from_peaks(self, max_num: int, width: int):
        """Gte the windows for the most brightest Bragg peaks."""
        self._check_attr("peaks")
        if self._peaks.shape[0] == 0:
            raise CrystalMapperError(
                "There is no peak found on the image. Please check your peaks table."
            )
        df = self._peaks.nlargest(max_num, "mass")
        self._windows = _create_windows_from_width(df, width)
        return

    def _calc_intensity_in_windows(self):
        """Get the intensity array as a function of index of frames."""
        self._check_attr("frames_arr")
        self._check_attr("windows")
        self._intensity = _track_peaks(self._frames_arr, self._windows)
        if self._dark is not None:
            self._bkg_intensity = _average_intensity(self._dark, self._windows)
            self._intensity = (
                self._intensity.transpose() - self._bkg_intensity
            ).transpose()
        else:
            _my_print("Attribute 'dark' is None. No background correction.")
        return

    def _assign_q_values(self) -> None:
        """Assign the values to the windows dataframe."""
        self._check_attr("ai")
        self._check_attr("windows")
        # change the unit of Q to inverse A
        qa = self._ai.qArray() / 10.0
        self._windows["Q"] = [qa[row.y, row.x] for row in self._windows.itertuples()]
        return

    def _assign_d_values(self) -> None:
        self._windows["d"] = 2.0 * math.pi / self._windows["Q"]
        return

    def _Q_to_twotheta(self, Q: np.ndarray):
        self._check_attr("ai")
        # change wavelength unit to A
        w = self._ai.wavelength * 1e10
        return np.rad2deg(2.0 * np.arcsin(Q * w / (4.0 * math.pi)))

    def _d_to_twotheta(self, d: np.ndarray):
        return self._Q_to_twotheta(2.0 * math.pi / d)

    def _assign_twotheta_values(self) -> None:
        # wavelength meter, Q inverse nanometer
        self._windows["twotheta"] = self._Q_to_twotheta(self._windows["Q"])
        return

    def _reshape_intensity(self) -> None:
        """Reshape the intensity array."""
        self._check_attr("metadata")
        self._check_attr("intensity")
        self._intensity = _reshape_to_ndarray(self._intensity, self._metadata)
        return

    def _calc_coords(self) -> None:
        """Calculate the coordinates."""
        self._check_attr("metadata")
        self._coords = _get_coords(self._metadata)
        return

    def _dark_to_xarray(self) -> xr.DataArray:
        """Convert the dark image to DataArray."""
        self._check_attr("dark")
        return xr.DataArray(self._dark, dims=["pixel_y", "pixel_x"])

    def _light_to_xarray(self) -> xr.DataArray:
        """Convert the light image to DataArray"""
        self._check_attr("light")
        return xr.DataArray(self._light, dims=["pixel_y", "pixel_x"])

    def _intensity_to_xarray(self) -> xr.DataArray:
        """Convert the intensity array to DataArray"""
        self._check_attr("intensity")
        dims = ["dim_{}".format(i) for i in range(self._intensity.ndim - 1)]
        arr = xr.DataArray(self._intensity, dims=["grain"] + dims)
        if self._coords is not None:
            coords = self._coords_to_dict()
            arr = arr.assign_coords(coords)
        return arr

    def _windows_to_xarray(self) -> xr.DataArray:
        """Convert the windows DataFrame to xarray."""
        self._check_attr("windows")
        return self._windows.rename_axis(index="grain").to_xarray()

    def _coords_to_dict(self) -> dict:
        """Convert the coordinates to dictionary."""
        self._check_attr("coords")
        return {"dim_{}".format(i): coord for i, coord in enumerate(self._coords)}

    def _dspacing_to_xarray(self) -> xr.DataArray:
        return xr.DataArray(
            self._dspacing, dims=["grain", "d_idx"], attrs={"units": r"nm$^{-1}$"}
        )

    def _hkl_to_xarray(self) -> xr.DataArray:
        return xr.DataArray(self._hkl, dims=["grain", "d_idx", "hkl_idx", "reciprocal"])

    def _n_hkl_to_xarray(self) -> xr.DataArray:
        return xr.DataArray(self._n_hkl, dims=["grain", "d_idx"])

    def _to_dataset(self) -> xr.Dataset:
        """Convert the calculation results to DataSet."""
        dct = dict()
        if self._dark is not None:
            dct["dark"] = self._dark_to_xarray()
        if self._light is not None:
            dct["light"] = self._light_to_xarray()
        if self._intensity is not None:
            dct["intensity"] = self._intensity_to_xarray()
        ds = xr.Dataset(dct)
        if self._windows is not None:
            ds2 = self._windows_to_xarray()
            ds = ds.merge(ds2)
        if self._dspacing is not None:
            ds = ds.assign({"dspacing": self._dspacing_to_xarray()})
        if self._hkl is not None:
            ds = ds.assign({"hkl": self._hkl_to_xarray()})
        if self._n_hkl is not None:
            ds = ds.assign({"n_hkl": self._n_hkl_to_xarray()})
        if self._metadata is not None:
            ds = ds.assign_attrs(**self._metadata)
        self._crystal_maps = ds
        return ds

    def _get_frame(self, index: int) -> xr.DataArray:
        """Get the frame of at the index."""
        self._check_attr("frames_arr")
        frames = self._frames_arr[index].compute()
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
        frame = self._get_frame(index)
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_windows_on_frame(self, index: int, *args, **kwargs) -> FacetGrid:
        """Show the windows on the frame at the index."""
        self._check_attr("windows")
        facet = self.show_frame(index, *args, **kwargs)
        _draw_windows(self._windows, facet.axes)
        return facet

    def show_dark(self, *args, **kwargs) -> FacetGrid:
        """Show the dark image."""
        self._check_attr("dark")
        frame = self._dark_to_xarray()
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_light(self, *args, **kwargs) -> FacetGrid:
        """Show the light image."""
        self._check_attr("light")
        frame = self._light_to_xarray()
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_light_sub_dark(self, *args, **kwargs) -> FacetGrid:
        """Show the dark subtracted light image."""
        self._check_attr("light")
        light = self._light_to_xarray()
        if self._dark is not None:
            dark = self._dark_to_xarray()
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
        _draw_windows(self._windows, facet.axes)
        return facet

    def show_intensity(self, **kwargs) -> FacetGrid:
        """Show the intensity array."""
        arr = self._intensity_to_xarray()
        return _auto_plot(arr, title=None, invert_y=True, **kwargs)

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

    def _try_calc_q_and_d(self) -> None:
        try:
            self._calc_coords()
        except CrystalMapperError as e:
            print(e)
        return

    def auto_process(self) -> None:
        """Automatically process the data in the standard protocol."""
        self.find_Bragg_spots()
        self.create_crystal_maps()
        return

    def tune_RoI(self, number: int, half_width: int) -> None:
        self._config.RoI_number = number
        self._config.RoI_half_width = half_width
        self._align_RoI()
        self.show_windows()
        return

    def create_crystal_maps(self) -> None:
        self._align_RoI()
        self._calc_intensity_in_windows()
        self._try_reshape_intensity()
        self._to_dataset()
        return

    def _align_RoI(self) -> None:
        self._calc_windows_from_peaks(
            self._config.RoI_number, self._config.RoI_half_width
        )
        return

    def find_Bragg_spots(self) -> None:
        self._squeeze_shape_and_extents()
        self._calc_dark_and_light_from_frames_arr(self._config.slice_of_frames)
        self._calc_peaks_from_dk_sub_frame(self._config.trackpy_kernel_size)
        self._try_calc_coords()
        self._try_calc_q_and_d()
        self._try_find_d_spacing(self._config.dspacing_bounds)
        return

    def _try_reshape_intensity(self):
        try:
            self._reshape_intensity()
        except CrystalMapperError as e:
            print(e)

    def _try_find_d_spacing(self, dspacing_tolerance):
        if dspacing_tolerance is not None:
            try:
                self._calc_hkls_in_a_range(*dspacing_tolerance)
            except CrystalMapperError as e:
                print(e)

    def _try_calc_q_and_d(self):
        try:
            self._assign_q_values()
            self._assign_d_values()
            self._assign_twotheta_values()
        except CrystalMapperError as e:
            print(e)

    def _try_calc_coords(self):
        try:
            self._calc_coords()
        except CrystalMapperError as e:
            print(e)

    def load(
        self,
        *,
        geometry: str = None,
        structure: str = None,
        crystal_maps: str = None,
        peak_index: str = None
    ) -> None:
        if geometry:
            self._load_ai(geometry)
        if structure:
            self._load_structure(structure)
        if crystal_maps:
            self._load_crystal_maps(crystal_maps)
        if peak_index:
            self._load_peak_index(peak_index)
        return

    def _load_ai(self, filename: str) -> None:
        ai = pyFAI.load(filename)
        self._ai = ai
        self._ubmatrix.geo = ai
        return

    def _load_frames_arr(self, filename: str) -> None:
        self._frames_arr = xr.load_dataarray(filename)
        return

    def _load_crystal_maps(self, filename: str) -> None:
        ds = xr.load_dataset(filename)
        self._crystal_maps = ds
        w_names = []
        for name in ds:
            if name not in self._window_names:
                self.__setattr__(str(name), ds[name].values)
            else:
                w_names.append(name)
        self._windows = ds[w_names].to_dataframe()
        try:
            self._assign_q_values()
            self._assign_d_values()
        except CrystalMapperError as error:
            print(error)
        return

    def _load_cell(self, filename: str) -> None:
        with pathlib.Path(filename).open("r") as f:
            dct = yaml.safe_load(f)
        self._cell = Cell(**dct)
        return

    def _load_cell_from_cif(self, cif_file: str) -> None:
        stru: Structure = loadStructure(cif_file, fmt="cif")
        lat: Lattice = stru.lattice
        self._set_cell_by_lat(lat)
        return

    def _set_cell_by_lat(self, lat: Lattice) -> None:
        self._cell = Cell(
            a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma
        )
        return

    def _load_peak_index(self, nc_file: str) -> None:
        self._peak_index = xr.load_dataset(nc_file)
        return

    def _load_structure(self, filename: str) -> None:
        stru: Structure = loadStructure(filename)
        self._stru = stru
        self._ubmatrix.lat = stru.lattice
        self._set_cell_by_lat(stru.lattice)
        return

    def load_bluesky_v1(self, run: BlueskyRun) -> None:
        self._metadata = dict(run.start)
        self._frames_arr = run.xarray_dask()[self._config.image_data_key]
        return

    def load_bluesky_v2(self, run: BlueskyRun) -> None:
        self._metadata = dict(run.metadata["start"])
        self._frames_arr = run.primary.to_dask()[self._config.image_data_key]
        return

    def save(
        self,
        *,
        geometry: str = None,
        structure: str = None,
        crystal_maps: str = None,
        peak_index: str = None
    ):
        if geometry:
            self._check_attr("ai")
            self._save_ai(geometry)
        if structure:
            self._check_attr("stru")
            self._save_structure(structure)
        if crystal_maps:
            self._check_attr("crystal_maps")
            self._save_crystal_maps(crystal_maps)
        if peak_index:
            self._check_attr("peak_index")
            self._save_peak_index(peak_index)
        return

    def _save_peak_index(self, filename: str) -> None:
        self._peak_index.to_netcdf(filename)
        return

    def _save_ai(self, poni_file: str) -> None:
        self._ai.save(poni_file)
        return

    def _save_crystal_maps(self, filename: str) -> None:
        self._crystal_maps.to_netcdf(filename)
        return

    def _save_structure(self, filename: str) -> None:
        self._stru.write(filename, "cif")
        return

    def _calc_hkls_in_a_range(self, lb: float, rb: float):
        """Calculate the hkl in a range of Q value."""
        self._check_attr("windows")
        self._check_attr("cell")
        if "d" not in self._windows.columns:
            if "Q" not in self._windows.columns:
                self._assign_q_values()
            self._assign_d_values()
        if self._all_dspacing is None:
            self._calc_ds_and_hkls()
        dspacings = []
        hkls = []
        n_hkls = []
        for d in self._windows["d"]:
            l_idx, r_idx = self._search_hkls_idx(d, lb, rb)
            dspacing = self._all_dspacing[l_idx:r_idx]
            dspacings.append(dspacing)
            hkl = self._all_hkl[l_idx:r_idx]
            hkls.append(hkl)
            n_hkl = self._all_n_hkl[l_idx:r_idx]
            n_hkls.append(n_hkl)
        self._dspacing = self._stack_arrays(dspacings)
        self._hkl = self._stack_arrays(hkls)
        self._n_hkl = self._stack_arrays(n_hkls)
        return

    def _calc_ds_and_hkls(self) -> None:
        dmin = self._windows["d"].min()
        dhkls = sorted(self._cell.d_spacing(dmin).values())
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
        self._all_dspacing = np.asarray(ds[::-1])
        self._all_twotheta = self._d_to_twotheta(self._all_dspacing)
        self._all_hkl = np.asarray(hkls[::-1])
        self._all_n_hkl = np.asarray(rows[::-1])
        return

    def _search_hkls_idx(
        self, d: float, lb: float, rb: float
    ) -> typing.Tuple[int, int]:
        ratio = np.divide(self._all_dspacing, d)
        return bisect.bisect(ratio, lb), bisect.bisect(ratio, rb)

    def _calc_hkls_lower_and_upper(self, two_theta_tolerance: float) -> None:
        """Calculate hkls and assign them to the peaks.

        Find the upper and lower bound of the Q for each Q value in the dataframe. The hkls that have the upper
        and lower bound values are the possible hkls for that peak. The index of the Q value is the index of the
        group of possible hkls. The index is recorded in the dataframe for both upper and lower bound.
        """
        dtt = two_theta_tolerance
        del two_theta_tolerance
        self._check_attr("windows")
        self._check_attr("cell")
        if "twotheta" not in self._windows.columns:
            if "Q" not in self._windows.columns:
                self._assign_q_values()
            self._assign_d_values()
            self._assign_twotheta_values()
        if self._all_dspacing is None:
            self._calc_ds_and_hkls()
        uppers, lowers = [], []
        for tt in self._windows["twotheta"]:
            l, u = self._search_bounds(tt - dtt, tt + dtt)
            uppers.append(u)
            lowers.append(l)
        self._windows["lower_idx"] = lowers
        self._windows["upper_idx"] = uppers
        return

    def _calc_diff_2theta(self) -> None:
        idxs = []
        diffs = []
        for tt, lower, upper in zip(
            self._windows["twotheta"],
            self._windows["lower_idx"],
            self._windows["upper_idx"],
        ):
            ltt = self._all_twotheta[lower] if lower >= 0 else float("-inf")
            utt = self._all_twotheta[upper] if upper >= 0 else float("inf")
            l_diff, u_diff = tt - ltt, utt - tt
            idx, diff = (lower, l_diff) if l_diff <= u_diff else (upper, u_diff)
            idxs.append(idx)
            diffs.append(diff)
        self._windows["closet_idx"] = idxs
        self._windows["diff"] = diffs
        return

    def _search_bounds(
        self, lower_tt: float, upper_tt: float
    ) -> typing.Tuple[int, int]:
        left = np.searchsorted(self._all_twotheta, lower_tt, side="left")
        right = max(np.searchsorted(self._all_twotheta, upper_tt, side="right") - 1, 0)
        # if no peaks found, use the closet
        if right < left:
            d1, d2 = (
                lower_tt - self._all_twotheta[right],
                upper_tt - self._all_twotheta[left],
            )
            if d1 < d2:
                left = right
            else:
                right = left
        return left, right

    def _load_lat_for_ubmatrix(self, cif_file: str):
        """Load the structure of the sample. The `cell` and `ubmatrix.lattice` will be loaded."""
        self._ubmatrix.set_lat_from_cif(cif_file)
        lat = self._ubmatrix.lat
        self._cell = Cell(
            a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma
        )
        return

    def _search_two_peaks(self, peaks: typing.List[int]) -> typing.Tuple[int, int]:
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
            "diff" not in self._windows.columns
            or "closet_idx" not in self._windows.columns
        ):
            raise CrystalMapperError("Please run calc_hkls first.")
        # get
        df = self._windows.loc[peaks]
        # run
        sel_df: pd.DataFrame = df.nsmallest(2, columns=["diff_dspacing"])
        if sel_df.shape[0] < 2:
            raise CrystalMapperError(
                "There are less than 2 peaks found. Please check the `windows` dataframe."
            )
        return tuple(sel_df.index.tolist())

    def _get_hkls(self, left: int, right: int):
        ns = self._all_n_hkl[left : right + 1]
        hklss = self._all_hkl[left : right + 1]
        return np.concatenate([hkls[:n] for n, hkls in zip(ns, hklss)])

    def _loss(self, hkls: np.ndarray) -> float:
        cost = self._get_losses(hkls)
        return np.min(cost)

    def _get_losses(self, hkls: np.ndarray) -> np.ndarray:
        rhkls = np.around(hkls)
        vs = self._ubmatrix.reci_to_cart(hkls)
        rvs = self._ubmatrix.reci_to_cart(rhkls)
        diffs_sq = np.sum((rvs - vs) ** 2, axis=1)
        lens_sq = np.sum(vs**2, axis=1)
        cost = np.sqrt(diffs_sq / lens_sq)
        return cost

    def _index_peaks(
        self, peak1: int, peak2: int, others: typing.List[int]
    ) -> typing.Generator[xr.Dataset, None, None]:
        """Use the hkls of two peaks to calculate U matrixs and use them to index other peaks. Return the hkls."""
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
        row1 = self._windows.loc[peak1]
        l1, r1 = int(row1["lower_idx"]), int(row1["upper_idx"])
        # a list of hkls, zero padded
        hkls1 = self._get_hkls(l1, r1)
        return hkls1

    def _set_us_for_peaks(self, peak1: int, peak2: int) -> None:
        row1 = self._windows.loc[peak1]
        row2 = self._windows.loc[peak2]
        xy1 = np.array([row1["x"], row1["y"]])
        xy2 = np.array([row2["x"], row2["y"]])
        self._ubmatrix.set_u1_from_xy(xy1)
        self._ubmatrix.set_u2_from_xy(xy2)
        return

    def _index_others(
        self, h1: np.ndarray, h2: np.ndarray, others: typing.List[int]
    ) -> xr.Dataset:
        self._ubmatrix.set_h1_from_hkl(h1)
        self._ubmatrix.set_h2_from_hkl(h2)
        self._ubmatrix.get_U()
        res = [h1, h2]
        for k in others:
            row = self._windows.loc[k]
            xy = np.array([row["x"], row["y"]])
            u = self._ubmatrix.xy_to_lab(xy)
            v = self._ubmatrix.lab_to_cart(u)
            hkl = self._ubmatrix.cart_to_reci(v)
            res.append(hkl)
        res = np.asarray(res)
        lval = self._loss(res[2:])
        ds = xr.Dataset({"hkls": (["peak", "hkl"], res), "loss": lval})
        return ds

    def _index_peaks_in_one_grain(
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
            yield from self._index_peaks(p1, p2, others)
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
        return _get_anlge(self._ubmatrix.u1, self._ubmatrix.u2)

    def _get_anlge_in_grain_frame(self) -> float:
        return _get_anlge(self._ubmatrix.h1, self._ubmatrix.h2)

    def _get_anlge_h1_h2(
        self, peak1: int, peak2: int
    ) -> typing.Generator[AngleComparsion, None, None]:
        hkls1 = self._get_hkls_for_peak(peak1)
        hkls2 = self._get_hkls_for_peak(peak2)
        self._set_us_for_peaks(peak1, peak2)
        angle0 = self._get_angle_in_sample_frame()
        n1 = hkls1.shape[0]
        n2 = hkls2.shape[0]
        for i in range(n1):
            for j in range(n2):
                self._ubmatrix.set_h1_from_hkl(hkls1[i])
                self._ubmatrix.set_h2_from_hkl(hkls2[j])
                angle = self._get_anlge_in_grain_frame()
                if 1e-8 < abs(angle) < (180.0 - 1e-8):
                    diff = abs(angle - angle0)
                    yield AngleComparsion(
                        self._ubmatrix.h1, self._ubmatrix.h2, angle0, angle, diff
                    )
        return

    def _get_hkl_for_a_peak(self, peak: int) -> HKL:
        row = self._windows.loc[peak]
        xy = np.array([row["x"], row["y"]])
        u = self._ubmatrix.xy_to_lab(xy)
        v = self._ubmatrix.lab_to_cart(u)
        hkl = self._ubmatrix.cart_to_reci(v)
        return hkl

    def _get_indexing_result_for_peaks(self, peaks: typing.List[int]) -> np.ndarray:
        return np.stack([self._get_hkl_for_a_peak(peak) for peak in peaks])

    def _get_U(self, h1: HKL, h2: HKL) -> Matrix:
        self._ubmatrix.h1 = h1
        self._ubmatrix.h2 = h2
        self._ubmatrix.get_U()
        return self._ubmatrix.U

    def _guess_miller_index(
        self,
        peaks: typing.List[int],
        first_n: T.Optional[int] = None,
        index_all: bool = True,
    ) -> None:
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
        if first_n is None:
            first_n = sys.maxsize
        N = len(peaks)
        # collect results
        lsts = defaultdict(list)
        all_peaks = self._windows.index.values if index_all else np.array(peaks)
        M = len(all_peaks)
        # fill in the results
        count = 0
        for i, j in itertools.permutations(range(N), 2):
            other = (all_peaks != peaks[i]) & (all_peaks != peaks[j])
            results = self._get_anlge_h1_h2(peaks[i], peaks[j])
            results: T.List[AngleComparsion] = _get_n_largest(results, first_n)
            for ac in results:
                u = self._get_U(ac.h1, ac.h2)
                hkls = self._get_indexing_result_for_peaks(all_peaks)
                losses = self._get_losses(hkls)
                loss = np.min(losses[other])
                lsts["U"].append(u)
                lsts["hkls"].append(hkls)
                lsts["losses"].append(losses)
                lsts["loss"].append(loss)
                lsts["angle_sample"].append(ac.angle_sample)
                lsts["angle_grain"].append(ac.angle_grain)
                lsts["diff_angle"].append(ac.diff_angle)
                lsts["peak1"].append(peaks[i])
                lsts["peak2"].append(peaks[j])
                count += 1
        if count == 0:
            raise CrystalMapperError(
                "No peaking indexing results were found. Please tune up the tth tolerance or checking the data."
            )
        # summarize the results
        self._peak_index = xr.Dataset(
            {
                "U": (["candidate", "dim_0", "dim_1"], lsts["U"]),
                "hkls": (["candidate", "peak", "dim_1"], lsts["hkls"]),
                "losses": (["candidate", "peak"], lsts["losses"]),
                "loss": (["candidate"], lsts["loss"]),
                "angle_sample": (["candidate"], lsts["angle_sample"], {"units": "deg"}),
                "angle_grain": (["candidate"], lsts["angle_grain"], {"units": "deg"}),
                "diff_angle": (["candidate"], lsts["diff_angle"], {"units": "deg"}),
                "peak1": (["candidate"], lsts["peak1"]),
                "peak2": (["candidate"], lsts["peak2"]),
            },
            {"peak": (["peak"], all_peaks)},
        )
        return

    def prepare_for_indexing(self) -> None:
        """Prepare for the peak indexing."""
        self._calc_hkls_lower_and_upper(self._config.index_tth_tolerance)
        return

    def guess_miller_index(self, peaks: T.List[int]) -> None:
        """Guess the index of the peaks in one grain.

        Parameters
        ----------
        peaks : typing.List[int]
            The index of the peaks in the table.
        """
        return self._guess_miller_index(
            peaks, self._config.index_best_n, self._config.index_all_peaks
        )

    def print_indexing_result(self, best_n: T.Optional[int] = None) -> None:
        """Print out the indexing results.

        Parameters
        ----------
        best_n : T.Optional[int], optional
            Only print out best n results, by default None
        """
        if not best_n:
            best_n = self._peak_index.sizes["candidate"]
        data = self._peak_index.sortby("loss").isel({"candidate": slice(0, best_n)})
        return self._print_indexing_result(data)

    def _print_indexing_result(self, data: xr.Dataset) -> None:
        n = data.sizes["candidate"]
        for i in range(n):
            sel = data.isel({"candidate": i})
            self._print_group(sel)
        return

    def _print_group(self, data: xr.Dataset) -> None:
        data = data.sortby("losses")
        # part 1
        print(
            "Use peak '{}' and peak '{}' in the indexing.".format(
                data["peak1"].item(), data["peak2"].item()
            )
        )
        header = self._get_header_df(data)
        print()
        print(_tableize(header))
        print()
        # part 2
        print("The U matrix is shown below.")
        print()
        print(_str_matrix(np.round(data["U"].data, 2)))
        print()
        # part 3
        print("Below is the prediction of the hkls.")
        print()
        body = self._get_body_df(data)
        print(_tableize(body))
        print()
        return

    def _get_body_df(self, data):
        n = data.sizes["peak"]
        lst1 = [""] * n
        lst2 = [""] * n
        lst3 = [""] * n
        lst4 = [0.0] * n
        for i in range(n):
            sel = data.isel({"peak": i})
            lst1[i] = sel["peak"].item()
            lst2[i] = "{:.2f}, {:.2f}, {:.2f}".format(*sel["hkls"].values)
            lst3[i] = "{:.0f}, {:.0f}, {:.0f}".format(*sel["hkls"].values)
            lst4[i] = np.round(sel["losses"].item() * 100.0, 1)
        df = pd.DataFrame(
            {
                "peak ID": lst1,
                "guessed hkl": lst2,
                "integer hkl": lst3,
                "indexing error [%]": lst4,
            }
        )
        return df

    def _get_header_df(self, data) -> pd.DataFrame:
        dct = defaultdict(list)
        dct["measured angle [deg]"].append("{:.2f}".format(data["angle_sample"].item()))
        dct["predicted angle [deg]"].append("{:.2f}".format(data["angle_grain"].item()))
        dct["min indexing error [%]"].append(
            "{:.1f}".format(data["loss"].item() * 100.0)
        )
        df = pd.DataFrame(dct)
        return df

    def show_crystal_maps(
        self, peaks: T.Optional[T.List[int]] = None, **kwargs
    ) -> None:
        """Show the crystal maps of certain peaks.

        Parameters
        ----------
        peaks : typing.List[int]
            A list of integer of the peaks.
        size: float
            The size of one cystal map.
        """
        return _show_crystal_maps(self._crystal_maps, peaks, **kwargs)
