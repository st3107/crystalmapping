import itertools
import math
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyFAI
import seaborn as sns
import xarray as xr
from diffpy.structure import Lattice, Structure, loadStructure
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import Cell
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from crystalmapping.baseobject import _auto_plot_dataset, _show_crystal_maps
from crystalmapping.preprocessor import EulerAngle, Preprocessor
from crystalmapping.ubmatrix import UBMatrix, _get_u_from_geo, _get_rot_matrix, _get_U_from_cart_and_inst

HKL = np.ndarray
Matrix = np.ndarray
T = typing
IDS = T.Union[T.List, int, slice]


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
class IndexResult(object):
    """The result of peak indexing."""

    peak1: int
    peak2: int
    u_mat: Matrix
    hkls: np.ndarray
    losses: np.ndarray
    loss: np.ndarray
    ac: AngleComparsion

    def __eq__(self, __o: object) -> bool:
        return self.loss == __o.loss

    def __lt__(self, __o: object) -> bool:
        return self.loss < __o.loss


def _to_slice(_ids: IDS) -> IDS:
    return slice(None, _ids) if isinstance(_ids, int) else _ids


def _choose_best_u_mats(peak_index: xr.Dataset, u_mat_ids: IDS) -> xr.Dataset:
    u_mat_ids = _to_slice(u_mat_ids)
    chosen = peak_index.sortby("loss")["candidate"][u_mat_ids]
    return chosen


def _choose_best_peaks(
    peak_index: xr.Dataset, u_mat_id: int, peak_ids: IDS
) -> xr.Dataset:
    peak_ids = _to_slice(peak_ids)
    chosen = peak_index.sel({"candidate": u_mat_id}).sortby("losses")["peak"][peak_ids]
    return chosen


def _str_matrix(data: Matrix) -> str:
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


def _normalize(v: np.ndarray) -> np.ndarray:
    nv = np.linalg.norm(v)
    if nv == 0.0:
        raise IndexError("length of vector is zero.")
    return v / nv


def _get_angle_in_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = _normalize(v1)
    v2_u = _normalize(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def _get_peaks_table(dataset: xr.Dataset, ai: AzimuthalIntegrator) -> pd.DataFrame:
    peaks = dataset.to_dataframe()
    qa = ai.qArray() / 10.0
    w = ai.wavelength * 1e10
    y = dataset["y"].data.astype(np.int64)
    x = dataset["x"].data.astype(np.int64)
    peaks["Q"] = np.array([qa[yy, xx] for yy, xx in zip(y, x)])
    peaks["d"] = 2.0 * math.pi / peaks["Q"]
    peaks["tth"] = np.rad2deg(2.0 * np.arcsin(peaks["Q"] * w / (4.0 * math.pi)))
    return peaks


def _load_datasets(
    data_files: List[str], euler_angles: List[EulerAngle]
) -> Preprocessor:
    pp = Preprocessor("grain", ["x", "y"])
    for f, a in zip(data_files, euler_angles):
        pp.load_data(f, *a)
    return pp


def _square_grid_subplots(n: int, size: float) -> Tuple[Figure, Sequence[Axes]]:
    ncol = int(np.round(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * size), squeeze=False)
    axes: Sequence[Axes] = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def _get_losses(ub: UBMatrix, hkls: np.ndarray) -> np.ndarray:
    rhkls = np.around(hkls)
    vs = ub.lat_to_grain(hkls)
    rvs = ub.lat_to_grain(rhkls)
    diffs_sq = np.sum((rvs - vs) ** 2, axis=1)
    lens_sq = np.sum(vs**2, axis=1)
    cost = np.sqrt(diffs_sq / lens_sq)
    return cost


def _make_peak_index(res: T.List[IndexResult], all_peaks: T.Any) -> xr.Dataset:
    return xr.Dataset(
        {
            "U": (["candidate", "dim_0", "dim_1"], [x.u_mat for x in res]),
            "hkls": (["candidate", "peak", "dim_1"], [x.hkls for x in res]),
            "losses": (["candidate", "peak"], [x.losses for x in res]),
            "loss": (["candidate"], [x.loss for x in res]),
            "angle_sample": (
                ["candidate"],
                [x.ac.angle_sample for x in res],
                {"units": "deg"},
            ),
            "angle_grain": (
                ["candidate"],
                [x.ac.angle_grain for x in res],
                {"units": "deg"},
            ),
            "diff_angle": (
                ["candidate"],
                [x.ac.diff_angle for x in res],
                {"units": "deg"},
            ),
            "peak1": (["candidate"], [x.peak1 for x in res]),
            "peak2": (["candidate"], [x.peak2 for x in res]),
        },
        {"peak": (["peak"], all_peaks)},
    ).sortby("loss")


def _get_hkl(ub: UBMatrix, row: pd.Series) -> np.ndarray:
    ub.set_R1(row["alpha"], row["beta"], row["gamma"])
    v_lab = ub.xy_to_lab(row[["x", "y"]].to_numpy())
    v_sample = ub.lab_to_sample_1(v_lab)
    v_grain = ub.sample_to_grain(v_sample)
    v_lat = ub.grain_to_lat_1(v_grain)
    return v_lat


def _index_using_U_matrix(
    peak_index: xr.Dataset, peaks: pd.DataFrame, ub: UBMatrix, idx: T.Any
) -> IndexResult:
    ub.U = peak_index["U"][idx].data
    return _get_index_result(peak_index, peaks, ub, idx)


def _get_index_result(peak_index, peaks, ub, idx):
    hkls = _get_hkls(peaks, ub)
    losses = _get_losses(ub, hkls)
    loss = np.min(losses)
    ac = AngleComparsion(
        None,
        None,
        peak_index["angle_sample"][idx].item(),
        peak_index["angle_grain"][idx].item(),
        peak_index["diff_angle"][idx].item(),
    )
    ir = IndexResult(
        peak_index["peak1"][idx].data,
        peak_index["peak2"][idx].data,
        ub.U,
        hkls,
        losses,
        loss,
        ac,
    )
    return ir


def _get_hkls(peaks, ub):
    hkls = np.array([_get_hkl(ub, row) for _, row in peaks.iterrows()])
    return hkls


def _index_peaks_by_U(
    peak_index: xr.Dataset, peaks: pd.DataFrame, ub: UBMatrix, u_mat_ids: T.List
) -> xr.Dataset:
    u_mat_ids = _choose_best_u_mats(peak_index, u_mat_ids)
    return _index_peaks(peak_index, peaks, ub, u_mat_ids)


def _index_peaks(
    peak_index: xr.Dataset, peaks: pd.DataFrame, ub: UBMatrix, u_mat_ids: T.List
) -> xr.Dataset:
    res = [_index_using_U_matrix(peak_index, peaks, ub, idx) for idx in u_mat_ids]
    return _make_peak_index(res, peaks.index)


def _fine_tune(
    ub: UBMatrix,
    peak_index: xr.Dataset,
    u_mat_ids: IDS,
    peaks: pd.DataFrame,
    peak_ids: IDS,
    **kwargs,
) -> xr.Dataset:
    kwargs.setdefault("xtol", 1e-4)
    kwargs.setdefault("gtol", 1e-4)
    kwargs.setdefault("ftol", 1e-4)
    u_mat_ids = _choose_best_u_mats(peak_index, u_mat_ids)
    for _id in u_mat_ids:
        _optmize_U_matrix(ub, peak_index, _id, peaks, peak_ids, **kwargs)
    return _index_peaks(peak_index, peaks, ub, u_mat_ids)


def _optmize_U_matrix(
    ub: UBMatrix,
    peak_index: xr.Dataset,
    u_mat_id: int,
    peaks: pd.DataFrame,
    peak_ids: IDS,
    **kwargs,
) -> None:
    peak_ids = _choose_best_peaks(peak_index, u_mat_id, peak_ids)
    sel_peaks = peaks.loc[peak_ids]

    def _chi(euler_angle: np.ndarray) -> np.ndarray:
        ub.set_U_by_euler_angle(euler_angle)
        hkls = _get_hkls(sel_peaks, ub)
        losses = _get_losses(ub, hkls)
        return losses

    ub.U = peak_index["U"][u_mat_id].data
    x0 = ub.get_euler_angles_from_U()
    res = least_squares(_chi, x0, **kwargs)
    ub.set_U_by_euler_angle(res.x)
    peak_index["U"][u_mat_id] = ub.U
    return


def _hist_error(peak_index: xr.Dataset, peak_ids: List[str] = None, size: float = 4.0, bins: Any = "auto") -> Figure:
    losses: pd.DataFrame = peak_index["losses"].to_dataframe()
    losses *= 100
    peak = (
        peak_index["peak"].to_numpy()
        if peak_ids is None
        else np.unique(np.array(peak_ids))
    )
    n = len(peak)
    fig, axes = _square_grid_subplots(n, size)
    for i in range(n):
        q = (
            "peak == '{}'".format(peak[i])
            if isinstance(peak[i], str)
            else "peak == {}".format(peak[i])
        )
        data = losses.query(q)
        sns.histplot(data, kde=True, ax=axes[i], bins=bins)
        axes[i].legend(
            [
                r"$\mu$ = {:.2f}, $\sigma$ = {:.3f}".format(
                    data["losses"].mean(), data["losses"].std()
                )
            ]
        )
        axes[i].set_title(r"Bragg Peak {}".format(peak[i]))
        axes[i].set_xlabel(r"Error (%)")
    fig.tight_layout()
    return fig


def _get_pair_angles(peak_index: xr.Dataset, u_mat_ids: IDS, peak_ids: IDS, ub: UBMatrix) -> T.List[xr.DataArray]:
    chosen_u_mat_ids = _choose_best_u_mats(peak_index, u_mat_ids)
    pair_angles = []
    for uid in chosen_u_mat_ids:
        pids = _choose_best_peaks(peak_index, uid, peak_ids)
        chosen_peak_index = peak_index.sel(dict(candidate=uid, peak=pids))
        pair_angle = _get_pair_angle(chosen_peak_index, ub)
        pair_angles.append(pair_angle)
    return pair_angles


def _get_pair_angle(chosen_peak_index: xr.Dataset, ub: UBMatrix) -> xr.Dataset:
    hkls: np.ndarray = chosen_peak_index["hkls"].data
    rhkls: np.ndarray = np.around(hkls, 0)
    v_grain = (ub.B @ hkls.T).T
    rv_grain = (ub.B @ rhkls.T).T
    n: int = hkls.shape[0]
    mat = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            a = _get_angle_in_deg(v_grain[i], v_grain[j])
            ar = _get_angle_in_deg(rv_grain[i], rv_grain[j])
            mat[j][i] = mat[i][j] = a - ar
    x = chosen_peak_index["peak"].data
    idx = np.arange(0, n, 1, dtype=np.uint64)
    data =  xr.Dataset(
        {"peak": (["peak"], x),
         "angle": (["dim_0", "dim_1"], mat)},
        {"dim_0": idx,
         "dim_1": idx}
    )
    data["dim_0"].attrs["standard_name"] = "peak"
    data["dim_1"].attrs["standard_name"] = "peak"
    data["angle"].attrs["standard_name"] = "$\Delta\Theta$"
    data["angle"].attrs["units"] = "deg"
    return data


def _matshow_pair_angles(pair_angles: T.List[xr.Dataset]) -> T.List[Figure]:
    return[_matshow_pair_angle(a) for a in pair_angles]


def _matshow_pair_angle(pair_angle: xr.Dataset) -> Figure:
    fig, ax = plt.subplots()
    x: list = pair_angle["peak"].data.tolist()
    n = len(x)
    loc = list(range(n))
    pair_angle["angle"].plot.imshow(ax=ax, infer_intervals=False, origin="upper")
    ax.set_aspect("equal")
    ax.set_xticks(loc)
    ax.set_xticklabels(x)
    ax.set_yticks(loc)
    ax.set_yticklabels(x)
    return fig


def _get_tqdm(Vs: List):
    n = len(Vs)
    count = 0
    for i in range(n):
        for j in range(n):
            for _ in range(len(Vs[i].T)):
                for _ in range(len(Vs[j].T)):
                    count += 1
    return iter(tqdm(range(count)))


def _guess_orientation(ubmatrix: UBMatrix, geo: AzimuthalIntegrator, chosen_peaks: pd.DataFrame, chosen_hkls: List[np.ndarray], error_in_deg: float) -> Tuple[List[Matrix], List[AngleComparsion], List, List]:
    n = chosen_peaks.shape[0]
    B = ubmatrix.B
    V5 = chosen_hkls
    index = chosen_peaks.index
    del chosen_hkls
    # get V2, list of vectors, the vectors are all vertical
    V1 = [_get_u_from_geo(row.x, row.y, geo).T for row in chosen_peaks.itertuples()]
    Rs = [_get_rot_matrix(row.alpha, row.beta, row.gamma) for row in chosen_peaks.itertuples()]
    V2 = [Rs[i].T @ V1[i] for i in range(n)]
    # get V4s, list of matrix of vertical vectors
    V4s = [B @ V5[i].T for i in range(n)]
    V3s = [Rs[i].T @ V4s[i] for i in range(n)]
    # get U matrix
    Us = []
    acs = []
    peaks1 = []
    peaks2 = []
    pbar = _get_tqdm(V3s)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u1 = V2[i].T
            u2 = V2[j].T
            a_sam = _get_angle_in_deg(u1, u2)
            V3i = V3s[i].T
            V3j = V3s[j].T
            for ii in range(len(V3i)):
                for jj in range(len(V3j)):
                    next(pbar)
                    h1 = V3i[ii]
                    h2 = V3j[jj]
                    a_gra = _get_angle_in_deg(h1, h2)
                    a_dif = abs(a_sam - a_gra)
                    if a_dif > error_in_deg:
                        continue
                    ac = AngleComparsion(h1, h2, a_sam, a_gra, a_dif)
                    U = _get_U_from_cart_and_inst(h1, h2, u1, u2)
                    Us.append(U)
                    acs.append(ac)
                    peaks1.append(index[i])
                    peaks2.append(index[j])
    if len(Us) == 0:
        raise IndexerError("Cannot find any valid vector pairs. Please increase the angle tolerance.")
    return Us, acs, peaks1, peaks2


def _guess_hkls(Us: List[Matrix], acs: List[AngleComparsion], peaks1: List, peaks2: List, ubmatrix: UBMatrix, geo: AzimuthalIntegrator, peaks: pd.DataFrame) -> List[IndexResult]:
    invB = ubmatrix.invB
    V1 = [_get_u_from_geo(row.x, row.y, geo).T for row in peaks.itertuples()]
    Rs = [_get_rot_matrix(row.alpha, row.beta, row.gamma) for row in peaks.itertuples()]
    irs = []
    for U, ac, peak1, peak2 in zip(Us, acs, peaks1, peaks2):
        hkls = []
        for v1, R in zip(V1, Rs):
            v5 = invB @ R @ U.T @ R.T @ v1
            hkls.append(v5.T)
        hkls = np.array(hkls)
        losses = _get_losses(ubmatrix, hkls)
        loss = np.min(losses)
        ir = IndexResult(peak1, peak2, U, hkls, losses, loss, ac)
        irs.append(ir)
    return irs


@dataclass
class IndexerConfig(object):
    """Configuration for the PeakIndexer."""

    dspacing_bounds: T.Optional[T.Tuple[float, float]] = None
    index_agl_tolerance: float = 1.0
    index_tth_tolerance: float = 0.1
    index_best_n: int = 256
    index_all_peaks: bool = True


class IndexerError(Exception):
    """Error of the PeakIndexer."""

    pass


@dataclass
class PredictedReflection(object):
    """The results of the predicted reflection."""

    q: T.Optional[np.ndarray] = None
    d: T.Optional[np.ndarray] = None
    tth: T.Optional[np.ndarray] = None
    hkls: T.Optional[T.List[np.ndarray]] = None


@dataclass
class PeakIndexer(object):
    """The Calculator of the crystal maps."""

    # config
    config: IndexerConfig
    # datasets
    _datasets: List[xr.Dataset] = field(default_factory=list)
    # pyFAI
    _ai: T.Optional[AzimuthalIntegrator] = None
    # cell
    _cell: T.Optional[Cell] = None
    # peak indexing results
    _peak_index: T.Optional[xr.Dataset] = None
    # peaks
    _peaks: T.Optional[pd.DataFrame] = None
    # UBmatrix object
    _ubmatrix: UBMatrix = UBMatrix()
    # cached value
    _pred: PredictedReflection = PredictedReflection()
    # previous result
    _previous_result: T.Optional[xr.Dataset] = None

    def load(
        self,
        data_files: List[str],
        euler_angles: List[EulerAngle],
        poni_file: str,
        stru_file: str,
    ) -> None:
        """Load the necessary data.

        Parameters
        ----------
        data_files : List[str]
            A list of netcdf4 files output by CrystalMapper.
        euler_angles : List[EulerAngle]
            A list of (alpha, beta, gamma) value of Euler anlge in YZY convention.
        poni_file : str
            A pyFAI poni file containing the geometry information of the experiment.
        stru_file : str
            A cif file containing the lattice information of the sample.
        """
        pp = _load_datasets(data_files, euler_angles)
        self._datasets = pp.data_lst
        self._load_ai(poni_file)
        self._load_structure(stru_file)
        merged = pp.process()
        self._peaks = _get_peaks_table(merged, self._ai)
        self._set_pred_reciprocal()
        return

    def load_miller_index(self, filename: str) -> None:
        """Load the guessing of Miller index.

        Parameters
        ----------
        filename : str
            A netcdf4 file of the guessing results.
        """
        self._peak_index = xr.load_dataset(filename).sortby("loss")
        return

    def load_previous_result(self, result_file: str) -> None:
        """Load the result from the previous peak indexing.

        Parameters
        ----------
        result_file : str
            Output file from the PeakIndexer.
        """
        self._previous_result = xr.load_dataset(result_file)
        return

    def save_miller_index(self, filename: str) -> None:
        """Save the guess of Miller index.

        Parameters
        ----------
        filename : str
            A destination of netcdf4 file.
        """
        self._peak_index.to_netcdf(filename)
        return

    def _set_pred_reciprocal(self) -> None:
        dmin = self._peaks["d"].min()
        dhkls = sorted(self._cell.d_spacing(dmin).values(), reverse=True)
        n = len(dhkls)
        if n == 0:
            raise IndexerError(
                "There is no matching d-spacing. Please check the cell attribute."
            )
        d = np.zeros((n,))
        hkls = [None] * n
        for i in range(n):
            d[i] = float(dhkls[i][0])
            hkls[i] = np.array(dhkls[i][1:])
        q = 2.0 * math.pi / d
        w = self._ai.wavelength * 1e10
        tth = np.rad2deg(2.0 * np.arcsin(q * w / (4.0 * math.pi)))
        self._pred.q = q
        self._pred.d = d
        self._pred.tth = tth
        self._pred.hkls = hkls
        return

    def _load_ai(self, filename: str) -> None:
        ai = pyFAI.load(filename)
        self._ai = ai
        self._ubmatrix.geo = ai
        return

    def _load_structure(self, filename: str) -> None:
        stru: Structure = loadStructure(filename)
        self._ubmatrix.lat = stru.lattice
        self._set_cell_by_lat(stru.lattice)
        return

    def _set_cell_by_lat(self, lat: Lattice) -> None:
        self._cell = Cell(
            a=lat.a, b=lat.b, c=lat.c, alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma
        )
        return

    def old_guess_miller_index(self, peak_ids: T.List[str]) -> None:
        """Guess the index of the peaks in one grain.

        Parameters
        ----------
        peaks : typing.List[str]
            The index of the peaks in the table.
        """
        for p in peak_ids:
            if p not in self._peaks.index:
                raise IndexerError(f"'{p}' is not a valid peak ID.")
        index_all = self.config.index_all_peaks
        best_n = self.config.index_best_n
        peak_ids: np.ndarray = np.unique(np.array(peak_ids))
        n = peak_ids.shape[0]
        vs4 = self._get_candidates(peak_ids)
        all_peaks = self._peaks.index.to_numpy() if index_all else peak_ids.copy()
        pairs = tqdm(itertools.permutations(range(n), 2), total=n * (n - 1))

        def gen():
            for i, j in pairs:
                other = (all_peaks != peak_ids[i]) & (all_peaks != peak_ids[j])
                results = self._get_anlge_h1_h2(
                    peak_ids[i], vs4[i], peak_ids[j], vs4[j]
                )
                for ac in results:
                    u = self._get_U(ac.h1, ac.h2)
                    hkls = self._get_indexing_result_for_peaks(all_peaks)
                    losses = self._get_losses(hkls)
                    loss = np.min(losses[other])
                    yield IndexResult(
                        peak1=peak_ids[i],
                        peak2=peak_ids[j],
                        u_mat=u,
                        hkls=hkls,
                        losses=losses,
                        loss=loss,
                        ac=ac,
                    )
            return

        res: T.List[IndexResult] = _get_n_largest(gen(), best_n)
        if len(res) == 0:
            raise IndexerError(
                "No peaking indexing results were found. Please tune up the tth or"
                "agl tolerance or checking the data."
            )
        # summarize the results
        self._peak_index = _make_peak_index(res, all_peaks)
        return

    def _get_candidates(self, peaks: T.List[str]) -> T.List[np.ndarray]:
        """Calculate hkls and assign them to the peaks.

        Find the upper and lower bound of the Q for each Q value in the dataframe. The hkls that have the upper
        and lower bound values are the possible hkls for that peak. The index of the Q value is the index of the
        group of possible hkls. The index is recorded in the dataframe for both upper and lower bound.
        """
        dtt = self.config.index_tth_tolerance
        candidates = []
        for p in peaks:
            tt = self._peaks.loc[p, "tth"]
            left = np.searchsorted(self._pred.tth, tt - dtt, side="left")
            right = np.searchsorted(self._pred.tth, tt + dtt, side="right")
            chosen = self._pred.hkls[left:right]
            if not chosen:
                raise IndexError(
                    "No candidate hkls found. Please increase the two theta tolerance."
                )
            hkls = np.concatenate(chosen)
            candidates.append(hkls)
        return candidates

    def _get_anlge_h1_h2(
        self, peak1: int, hkls1: np.ndarray, peak2: int, hkls2: np.ndarray
    ) -> typing.Generator[AngleComparsion, None, None]:
        self._set_vec_in_sample_frame(peak1, peak2)
        angle_in_sample = self._get_angle_in_sample_frame()
        if not (1e-8 < abs(angle_in_sample) < (180.0 - 1e-8)):
            return
        n1 = hkls1.shape[0]
        n2 = hkls2.shape[0]
        for i in range(n1):
            for j in range(n2):
                self._ubmatrix.set_h1_from_hkl(hkls1[i])
                self._ubmatrix.set_h2_from_hkl(hkls2[j])
                angle_in_grain = self._get_anlge_in_grain_frame()
                if not (1e-8 < abs(angle_in_grain) < (180.0 - 1e-8)):
                    continue
                diff = abs(angle_in_grain - angle_in_sample)
                if diff > self.config.index_agl_tolerance:
                    continue
                yield AngleComparsion(
                    self._ubmatrix.h1,
                    self._ubmatrix.h2,
                    angle_in_sample,
                    angle_in_grain,
                    diff,
                )
        return

    def _set_vec_in_sample_frame(self, peak1: int, peak2: int) -> None:
        row1 = self._peaks.loc[peak1]
        row2 = self._peaks.loc[peak2]
        xy1 = np.array([row1["x"], row1["y"]])
        xy2 = np.array([row2["x"], row2["y"]])
        self._ubmatrix.set_R1(row1["alpha"], row1["beta"], row1["gamma"])
        self._ubmatrix.set_R2(row2["alpha"], row2["beta"], row2["gamma"])
        self._ubmatrix.set_u1_from_xy(xy1)
        self._ubmatrix.set_u2_from_xy(xy2)
        return

    def _get_angle_in_sample_frame(self) -> float:
        return _get_angle_in_deg(self._ubmatrix.u1, self._ubmatrix.u2)

    def _get_anlge_in_grain_frame(self) -> float:
        return _get_angle_in_deg(self._ubmatrix.h1, self._ubmatrix.h2)

    def _get_U(self, h1: HKL, h2: HKL) -> Matrix:
        self._ubmatrix.h1 = h1
        self._ubmatrix.h2 = h2
        self._ubmatrix.calc_and_set_U()
        return self._ubmatrix.U

    def _get_losses(self, hkls: np.ndarray) -> np.ndarray:
        return _get_losses(self._ubmatrix, hkls)

    def _get_indexing_result_for_peaks(self, peaks: typing.List[str]) -> np.ndarray:
        return np.stack([self._get_hkl_for_a_peak(peak) for peak in peaks])

    def _get_hkl_for_a_peak(self, peak: int) -> HKL:
        row = self._peaks.loc[peak]
        xy = np.array([row["x"], row["y"]])
        self._ubmatrix.set_R1(row["alpha"], row["beta"], row["gamma"])
        v_lab = self._ubmatrix.xy_to_lab(xy)
        v_sample = self._ubmatrix.lab_to_sample_1(v_lab)
        v_grain = self._ubmatrix.sample_to_grain(v_sample)
        v_lat = self._ubmatrix.grain_to_lat_1(v_grain)
        return v_lat

    def show(self, u_mat_ids: IDS = 1, peak_ids: IDS = 10) -> None:
        """Print out the indexing results.

        Parameters
        ----------
        peak_ids : IDS, optional
            Only print out best n results, by default None
        """
        data = self._peak_index
        chosen_ids = _choose_best_u_mats(data, u_mat_ids)
        for _id in chosen_ids:
            self._print_data(data, _id, peak_ids)
        return

    def _print_data(self, data: xr.Dataset, u_mat_id: int, peak_ids: IDS) -> None:
        chosen = _choose_best_peaks(data, u_mat_id, peak_ids)
        data = data.sel({"candidate": u_mat_id, "peak": chosen})
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
        print(_str_matrix(np.round(data["U"].data, 3)))
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

    def visualize(self, dataset_id: int, peak_ids: List[str] = None, **kwargs) -> Figure:
        """Visualize the crystal maps.

        Parameters
        ----------
        dataset_id : int
            A 0-index ID of the dataset.
        peaks : List[str], optional
            A list of peak id in that dataset, by default None
        """
        if peak_ids is None:
            return _auto_plot_dataset(self._datasets[dataset_id], **kwargs)
        return _show_crystal_maps(self._datasets[dataset_id], peak_ids, **kwargs)

    def hist_error(
        self, peak_ids: List[str] = None, size: float = 4.0, bins: Any = "auto"
    ) -> Figure:
        """Plot the histogram of erros.

        Parameters
        ----------
        peak_ids : List[str], optional
            A list of peak IDs to plot, by default None, plot all peaks.
        size : float, optional
            Size in inches for the individual panel, by default 4.
        """
        return _hist_error(self._peak_index, peak_ids, size, bins)

    def index_peaks_by_U(self, u_mat_ids: IDS) -> None:
        """Index the peaks using the U matrix from previous results.

        Parameters
        ----------
        u_mat_ids : IDS
            0-index of the U matrix to use. It is from best to worst in previous result.
        """
        self._peak_index = _index_peaks_by_U(
            self._previous_result, self._peaks, self._ubmatrix, u_mat_ids
        )
        return

    def fine_tune(self, u_mat_ids: IDS, peak_ids: IDS, **kwargs) -> None:
        """Fine tune the U matrix.

        The fine tune is to minimize the sum of squares of the indexing errors of a set of peaks. The starting point of U is chosen from the previous indexing result from `guess_miller_index`. The fine tune is done in place so the peak_index attribute will be changed.

        Parameters
        ----------
        u_mat_ids : IDS
            Number or a list of numbers of U matrix to fine tune. If it is int, e. g. 3, tune the best 3 matrix. If it is a list of ints, e. g. [1, 4, 7], tune the No. 1, 4, 7 matrix.
        peak_ids : IDS
            Number of a list of numbers of peaks to minimize the indexing errors. If it is int, e. g. 6, minimize the best 6 peaks. If it is a list of ints, e. g. [1, 4, 8, 9], minimize No. 1, 4, 8, 9.
        """
        self._peak_index = _fine_tune(
            self._ubmatrix, self._peak_index, u_mat_ids, self._peaks, peak_ids, **kwargs
        )
        return

    def get_pair_angles(self, u_mat_ids: IDS, peak_ids: IDS) -> T.List[xr.DataArray]:
        """Obtain the angle difference matrix.

        The angle difference matrix is a matrix $\Theta$, where each element (i, j) is the the angle between peak i and peak j in grain frame by that in sample frame.

        $$
        \Theta(i, j) = \left< i, j \right>_{grain} - \left< i, j \right>_{sample}
        $$
        
        One U matrix will give us an angle difference matrix. The dimension of the matrix is equal to number of peaks chosen.

        Parameters
        ----------
        u_mat_ids : IDS
            Number or index of the chosen U matrix.
        peak_ids : IDS
            Number or index of the chosen peaks.

        Returns
        -------
        T.List[xr.DataArray]
            List of the angle difference matrix.
        """
        return _get_pair_angles(self._peak_index, u_mat_ids, peak_ids, self._ubmatrix)
    
    def matshow_pair_angles(self, pair_angles: T.List[xr.DataArray]) -> T.List[Figure]:
        """Visualize a list of angle difference matrix.
        
        The angle difference matrix is a matrix $\Theta$, where each element (i, j) is the the angle between peak i and peak j in grain frame by that in sample frame.

        $$
        \Theta(i, j) = \left< i, j \right>_{grain} - \left< i, j \right>_{sample}
        $$
        
        One U matrix will give us an angle difference matrix. The dimension of the matrix is equal to number of peaks chosen.

        Parameters
        ----------
        pair_angles : T.List[xr.DataArray]
            List of angle difference matrix

        Returns
        -------
        T.List[Figure]
            List of figures.
        """
        return _matshow_pair_angles(pair_angles)


    def guess_miller_index(self, peak_ids: T.List[str]) -> None:
        """Guess the index of the peaks in one grain.

        Parameters
        ----------
        peaks : typing.List[str]
            The index of the peaks in the table.
        """
        peak_ids: np.ndarray = np.unique(np.array(peak_ids))
        for p in peak_ids:
            if p not in self._peaks.index:
                raise IndexerError(f"'{p}' is not a valid peak ID.")
        index_all = self.config.index_all_peaks
        error_in_deg = self.config.index_agl_tolerance
        chosen_peaks = self._peaks.loc[peak_ids]
        chosen_hkls = self._get_candidates(peak_ids)
        all_peaks_ids = self._peaks.index if index_all else peak_ids
        all_peaks = self._peaks.loc[all_peaks_ids]
        Us, acs, peaks1, peaks2 = _guess_orientation(self._ubmatrix, self._ai, chosen_peaks, chosen_hkls, error_in_deg)
        irs = _guess_hkls(Us, acs, peaks1, peaks2, self._ubmatrix, self._ai, all_peaks)
        self._peak_index = _make_peak_index(irs, all_peaks_ids)
        best_n = self.config.index_best_n
        if best_n:
            self._peak_index = self._peak_index.sortby("loss").isel({"candidate": slice(None, best_n)})
        return
