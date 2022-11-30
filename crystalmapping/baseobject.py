import typing
from dataclasses import dataclass

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray.plot import FacetGrid


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
    dims = [k for k in da.sizes.keys() if k != "grain"]
    sizes = [da.sizes[d] for d in dims]
    kwargs.setdefault("col", "grain")
    kwargs.setdefault("col_wrap", 20)
    kwargs.setdefault("sharex", False)
    kwargs.setdefault("sharey", False)
    kwargs.setdefault("add_colorbar", False)
    vmin, vmax = limit_func(da)
    kwargs.setdefault("vmax", vmax)
    kwargs.setdefault("vmin", vmin)
    kwargs.setdefault("size", 5.0)
    kwargs.setdefault("aspect", sizes[1] / sizes[0])
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


def _auto_plot_dataset(
    ds: xr.Dataset,
    key: str = "intensity",
    title: typing.Tuple[str, str] = ("grain", "peak {}"),
    invert_y: bool = True,
    **kwargs
) -> Figure:
    facet = _auto_plot(ds[key], title=None, invert_y=invert_y, **kwargs)
    if title is not None:
        v_name, f_title = title
        vals: np.ndarray = ds[v_name].values
        axes: typing.List[plt.Axes] = facet.axes.flatten()
        for ax, val in zip(axes, vals):
            ax.set_title(f_title.format(val))
    facet.fig.tight_layout()
    return facet.fig


def _show_crystal_maps(
    data: xr.Dataset, peaks: typing.Optional[typing.List[int]], **kwargs
) -> Figure:
    if peaks is not None:
        data = data.sel({"grain": peaks})
    return _auto_plot_dataset(data, **kwargs)


@dataclass
class Config(object):

    enable_tqdm: bool = True


class Error(Exception):
    pass


@dataclass
class BaseObject(object):
    """The Calculator of the crystal maps.
    """

    # configuration
    _config: typing.Optional[Config] = None
    # crystal maps
    _dataset: typing.Optional[xr.Dataset] = None

    def load_dataset(self, filename: str) -> None:
        self._dataset = xr.load_dataset(filename)
        return

    def save_dataset(self, filename: str) -> None:
        self._dataset.to_netcdf(filename)
        return

    def visualize(
        self, peaks: typing.Optional[typing.List[int]] = None, **kwargs
    ) -> None:
        """Show the crystal maps of certain peaks.

        Parameters
        ----------
        peaks : typing.List[int]
            A list of integer of the peaks.
        """
        return _show_crystal_maps(self._dataset, peaks, **kwargs)
