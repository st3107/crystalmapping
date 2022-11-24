import logging
import math
import typing
from dataclasses import dataclass

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import xarray as xr
from tqdm.auto import tqdm
from xarray.plot import FacetGrid

from .baseobject import BaseObject, Config, Error, _auto_plot, _set_real_aspect

COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
T = typing
BlueskyRun = typing.Any
DataTuple = typing.Tuple[dict, xr.DataArray]


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
        raise MapperError("frame must have no less than 2 dimensions.")
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


def _track_peaks(
    frames: xr.DataArray, windows: pd.DataFrame, enable_tqdm: bool
) -> np.ndarray:
    """Create a list of tasks to compute the grain maps. Each task is one grain map."""
    # create intensity vs time for each grain
    intensities = []
    n = frames.shape[0]
    iter_range = tqdm(range(n), disable=(not enable_tqdm))
    for i in iter_range:
        intensity = _average_intensity(frames[i].values, windows)
        intensities.append(intensity)
    # axis: grain, frame
    return np.stack(intensities).transpose()


def _reshape_to_ndarray(arr: np.ndarray, metadata: dict) -> np.ndarray:
    if "shape" not in metadata:
        raise MapperError("Missing key '{}' in metadata.".format("shape"))
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
            raise MapperError("snaking only works for the 2 dimension array.")
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


def _min_and_max_along_time(
    data: xr.DataArray, enable_tqdm: bool
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Extract the minimum and maximum values of each pixel in a series of mean frames. Return a data array.
    First is the min array and the second is the max array."""
    min_arr = max_arr = np.mean(data[0].values, axis=0)
    iter_range = tqdm(range(1, len(data)), disable=(not enable_tqdm))
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



def _split_xy(run: BlueskyRun, image_name: str, phi_name: str, xy_names: T.List[str], decimal: int) -> T.List[DataTuple]:
    data = _get_data(run)
    dtups = _split_data(data, image_name, phi_name, xy_names, decimal)
    return dtups


def _get_data(run: BlueskyRun) -> xr.Dataset:
    if hasattr(run, "xarray_dask"):
        try:
            return run.xarray_dask()
        except Exception:
            return run.xarray()
    if hasattr(run, "primary") and hasattr(run.primary, "to_dask"):
        try:
            return run.primary.to_dask()
        except Exception:
            return run.primary.read()
    raise MapperError("{} is not a bluesky run".format(run))


def _split_data(data: xr.Dataset, image_name: str, phi_name: str, xy_names: T.List[str], decimal: int) -> T.List[DataTuple]:
    for name in xy_names:
        data[name] = np.round(data[name], decimal)
    grouped = data.set_index(time=xy_names).groupby("time")
    arrs = [d.sortby(phi_name) for _, d in grouped]
    dtups = [(_make_metadata(a, phi_name), a[image_name]) for a in arrs]
    return dtups


def _make_metadata(data: xr.Dataset, phi_name: str) -> dict:
    return {
        "shape": [len(data[phi_name])],
        "extents": [(data[phi_name].data.min(), data[phi_name].data.max())],
        "snaking": (False,)
    }


def _load_data_v1(run: BlueskyRun, image_data_key: str) -> DataTuple:
    metadata = dict(run.start)
    try:
        frames_arr = run.xarray_dask()[image_data_key]
    except Exception:
        frames_arr = run.xarray()[image_data_key]
    return metadata, frames_arr


def _load_data_v2(run: BlueskyRun, image_data_key: str) -> DataTuple:
    metadata = dict(run.metadata["start"])
    try:
        frames_arr = run.primary.to_dask()[image_data_key]
    except Exception:
        frames_arr = run.primary.read()[image_data_key]
    return metadata, frames_arr


@dataclass
class MapperConfig(Config):

    image_data_key: str = "dexela_image"
    RoI_number: int = 100
    RoI_half_width: int = 25
    trackpy_kernel_size: int = 25
    slice_of_frames: T.Optional[slice] = None


class MapperError(Error):
    pass


@dataclass
class CrystalMapper(BaseObject):
    """The Calculator of the crystal maps."""

    # configuration
    _config: T.Optional[MapperConfig] = None
    # dims: time, ..., pixel_y, pixel_x
    _frames_arr: T.Optional[xr.DataArray] = None
    # dims: pixel_y, pixel_x
    _dark: T.Optional[np.ndarray] = None
    # dims: pixel_y, pixel_x
    _light: T.Optional[np.ndarray] = None
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
    # dims: grain, d_idx, hkl_idx ,hkl
    _window_names = frozenset(["x", "y", "dx", "dy"])

    def _squeeze_shape_and_extents(self) -> None:
        """Squeeze the shape and extents so that it only has the dimension with length > 1."""
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
        frames_arr = (
            self._frames_arr[index_range]
            if index_range is not None
            else self._frames_arr
        )
        self._dark, self._light = _min_and_max_along_time(
            frames_arr, self._config.enable_tqdm
        )
        return

    def _calc_peaks_from_dk_sub_frame(
        self, radius: typing.Union[int, tuple], *args, **kwargs
    ):
        """Get the Bragg peaks on the light frame."""
        light = self._light if self._dark is None else self._light - self._dark
        self._peaks = tp.locate(light, 2 * radius + 1, *args, **kwargs)
        return

    def _calc_windows_from_peaks(self, max_num: int, width: int):
        """Gte the windows for the most brightest Bragg peaks."""
        if self._peaks.shape[0] == 0:
            raise MapperError(
                "There is no peak found on the image. Please check your peaks table."
            )
        df = self._peaks.nlargest(max_num, "mass")
        self._windows = _create_windows_from_width(df, width)
        return

    def _calc_intensity_in_windows(self):
        """Get the intensity array as a function of index of frames."""
        self._intensity = _track_peaks(
            self._frames_arr, self._windows, self._config.enable_tqdm
        )
        if self._dark is not None:
            self._bkg_intensity = _average_intensity(self._dark, self._windows)
            self._intensity = (
                self._intensity.transpose() - self._bkg_intensity
            ).transpose()
        else:
            logging.error("Attribute 'dark' is None. No background correction.")
        return

    def _reshape_intensity(self) -> None:
        """Reshape the intensity array."""
        self._intensity = _reshape_to_ndarray(self._intensity, self._metadata)
        return

    def _calc_coords(self) -> None:
        """Calculate the coordinates."""
        self._coords = _get_coords(self._metadata)
        return

    def _dark_to_xarray(self) -> xr.DataArray:
        """Convert the dark image to DataArray."""
        return xr.DataArray(self._dark, dims=["pixel_y", "pixel_x"])

    def _light_to_xarray(self) -> xr.DataArray:
        """Convert the light image to DataArray"""
        return xr.DataArray(self._light, dims=["pixel_y", "pixel_x"])

    def _intensity_to_xarray(self) -> xr.DataArray:
        """Convert the intensity array to DataArray"""
        dims = ["dim_{}".format(i) for i in range(self._intensity.ndim - 1)]
        arr = xr.DataArray(self._intensity, dims=["grain"] + dims)
        if self._coords is not None:
            coords = self._coords_to_dict()
            arr = arr.assign_coords(coords)
        return arr

    def _windows_to_xarray(self) -> xr.DataArray:
        """Convert the windows DataFrame to xarray."""
        return self._windows.rename_axis(index="grain").to_xarray()

    def _coords_to_dict(self) -> dict:
        """Convert the coordinates to dictionary."""
        return {"dim_{}".format(i): coord for i, coord in enumerate(self._coords)}

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
        self._dataset = ds
        return ds

    def _from_dataset(self, dataset: xr.Dataset) -> None:
        if "dark" in dataset:
            self._dark = dataset["dark"].data
        if "light" in dataset:
            self._light = dataset["light"].data
        if "intensity" in dataset:
            self._intensity = dataset["intensity"].data
        names = [w for w in self._window_names if w in dataset]
        if names:
            self._windows = self._dataset[names].to_dataframe()
        return

    def _get_frame(self, index: int) -> xr.DataArray:
        """Get the frame of at the index."""
        frames = self._frames_arr[index].compute()
        if frames.ndim == 3:
            return frames.mean(axis=0)
        elif frames.ndim == 2:
            return frames
        else:
            raise MapperError(
                "The dimension of the frame is {}. Require 2 or 3.".format(frames.ndim)
            )

    def show_frame(self, index: int, *args, **kwargs) -> FacetGrid:
        """Show the frame at that index."""
        frame = self._get_frame(index)
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_windows_on_frame(self, index: int, *args, **kwargs) -> FacetGrid:
        """Show the windows on the frame at the index."""
        facet = self.show_frame(index, *args, **kwargs)
        _draw_windows(self._windows, facet.axes)
        return facet

    def show_dark(self, *args, **kwargs) -> FacetGrid:
        """Show the dark image."""
        frame = self._dark_to_xarray()
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_light(self, *args, **kwargs) -> FacetGrid:
        """Show the light image."""
        frame = self._light_to_xarray()
        _set_vlim(kwargs, frame, 4.0)
        facet = frame.plot.imshow(*args, **kwargs)
        _set_real_aspect(facet.axes)
        return facet

    def show_light_sub_dark(self, *args, **kwargs) -> FacetGrid:
        """Show the dark subtracted light image."""
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
        facet = self.show_light_sub_dark(*args, **kwargs)
        _draw_windows(self._windows, facet.axes)
        return facet

    def show_intensity(self, **kwargs) -> FacetGrid:
        """Show the intensity array."""
        arr = self._intensity_to_xarray()
        return _auto_plot(arr, title=None, invert_y=True, **kwargs)

    def auto_process(self) -> None:
        """Automatically process the data in the standard protocol."""
        self.find_bragg_spots()
        self.create_crystal_maps()
        return

    def tune_RoI(self, number: int, half_width: int) -> None:
        """Tune the RoI number and half width.

        Parameters
        ----------
        number : int
            Number of the RoI regions. Choose from the strongest.
        half_width : int
            Number of pixels excluding the center for the half width of the RoI region.
        """
        self._config.RoI_number = number
        self._config.RoI_half_width = half_width
        self._align_roi()
        self.show_windows()
        return

    def find_bragg_spots(self) -> None:
        """Find the Bragg spots before creating the crystal maps."""
        self._squeeze_shape_and_extents()
        self._calc_dark_and_light_from_frames_arr(self._config.slice_of_frames)
        self._calc_peaks_from_dk_sub_frame(self._config.trackpy_kernel_size)
        self._calc_coords()
        return

    def create_crystal_maps(self) -> None:
        """Create the crystal maps after finding the Bragg spots."""
        self._align_roi()
        self._calc_intensity_in_windows()
        self._reshape_intensity()
        self._to_dataset()
        return

    def _align_roi(self) -> None:
        self._calc_windows_from_peaks(
            self._config.RoI_number, self._config.RoI_half_width
        )
        return

    def load_bluesky_v1(self, run: BlueskyRun) -> None:
        """Load the data and metadata from the version 1 databroker.

        Parameters
        ----------
        run : BlueskyRun
            A version 1 BlueskyRun (Header).
        """
        self._metadata, self._frames_arr = _load_data_v1(run, self._config.image_data_key)
        return

    def load_bluesky_v2(self, run: BlueskyRun) -> None:
        """Load the data and metadata from the version 2 databroker.

        Parameters
        ----------
        run : BlueskyRun
            A version 2 BlueskyRun.
        """
        self._metadata, self._frames_arr = _load_data_v2(run, self._config.image_data_key)
        return

    def load_data_tuple(self, data_tuple: DataTuple) -> None:
        """Load a tuple of (metadata, data) from splitted data.

        Parameters
        ----------
        data_tuple : DataTuple
            Tuple of (metadata, data).
        """
        self._metadata, self._frames_arr = data_tuple
        return
    
    def split_xy(self, run: BlueskyRun, phi_name: str, xy_names: T.List[str], decimal: int) -> T.List[DataTuple]:
        """Split the data by rounded (x, y) positions.

        Parameters
        ----------
        run : BlueskyRun
            Bluesky run object.
        phi_name : str
            Name of the rocking angle.
        xy_names : T.List[str]
            List of names of the x, y motors.
        decimal : int
            Decimal of x, y values to round to when grouping.

        Returns
        -------
        T.List[DataTuple]
            List of (metadata, data) tuple. Load by `load_data_tuple`.
        """
        return _split_xy(run, self._config.image_data_key, phi_name, xy_names, decimal)

    def visualize(
        self, peaks: typing.Optional[typing.List[int]] = None, **kwargs
    ) -> None:
        """Show the crystal maps of certain peaks.

        Parameters
        ----------
        peaks : typing.List[int]
            A list of integer of the peaks.
        """
        return super().visualize(peaks, **kwargs)

    def load_dataset(self, data_file: str) -> None:
        super().load_dataset(data_file)
        self._from_dataset(self._dataset)
        return
