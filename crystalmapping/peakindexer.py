import itertools
import math
import sys
import typing
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import pandas as pd
import pyFAI
import xarray as xr
from diffpy.structure import Lattice, Structure, loadStructure
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import Cell
from tqdm.auto import tqdm

from .baseobject import BaseObject, Config, Error
from .ubmatrix import UBMatrix

HKL = np.ndarray
Matrix = np.ndarray
T = typing


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


def _get_anlge(v1: np.ndarray, v2: np.ndarray) -> float:
    inner = np.dot(v1, v2)
    inner /= np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(inner))


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
class IndexerConfig(Config):

    dspacing_bounds: T.Optional[T.Tuple[float, float]] = None
    index_tth_tolerance: float = 0.1
    index_best_n: int = 3
    index_all_peaks: bool = True


class IndexerError(Error):
    pass


@dataclass
class PredictedReflection:

    q: T.Optional[np.ndarray] = None
    d: T.Optional[np.ndarray] = None
    tth: T.Optional[np.ndarray] = None
    hkls: T.Optional[T.List[np.ndarray]] = None


@dataclass
class PeakIndexer(BaseObject):
    """The Calculator of the crystal maps."""

    # config
    _config: IndexerConfig
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

    def load(self, data_file: str, poni_file: str, stru_file: str) -> None:
        self.load_dataset(data_file)
        self._load_ai(poni_file)
        self._load_structure(stru_file)
        self._set_actual_reciprocal()
        self._set_pred_reciprocal()
        return

    def load_miller_index(self, filename: str) -> None:
        self._peak_index = xr.load_dataset(filename)
        return

    def save_miller_index(self, filename: str) -> None:
        self._peak_index.to_netcdf(filename)
        return

    def _set_actual_reciprocal(self) -> np.ndarray:
        """Assign the values to the windows dataframe."""
        # change the unit of Q to inverse A
        self._peaks = self._dataset[["x", "y"]].to_dataframe()
        qa = self._ai.qArray() / 10.0
        w = self._ai.wavelength * 1e10
        y = self._dataset["y"].data
        x = self._dataset["x"].data
        self._peaks["Q"] = np.array([qa[yy, xx] for yy, xx in zip(y, x)])
        self._peaks["d"] = 2.0 * math.pi / self._peaks["Q"]
        self._peaks["tth"] = np.rad2deg(
            2.0 * np.arcsin(self._peaks["Q"] * w / (4.0 * math.pi))
        )
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

    def guess_miller_index(self, peaks: T.List[int]) -> None:
        """Guess the index of the peaks in one grain.

        Parameters
        ----------
        peaks : typing.List[int]
            The index of the peaks in the table.
        """
        first_n = self._config.index_best_n
        index_all = self._config.index_all_peaks
        if first_n is None:
            first_n = sys.maxsize
        peaks = np.array(peaks)
        n = peaks.shape[0]
        # choose candidates
        candidates = self._get_candidates(peaks)
        # collect results
        lsts = defaultdict(list)
        all_peaks = self._dataset["grain"].data if index_all else peaks.copy()
        # fill in the results
        count = 0
        for i, j in itertools.permutations(range(n), 2):
            other = (all_peaks != peaks[i]) & (all_peaks != peaks[j])
            results = self._get_anlge_h1_h2(
                peaks[i], candidates[i], peaks[j], candidates[j]
            )
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
            raise IndexerError(
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

    def _get_candidates(self, peaks: T.List[int]) -> T.List[np.ndarray]:
        """Calculate hkls and assign them to the peaks.

        Find the upper and lower bound of the Q for each Q value in the dataframe. The hkls that have the upper
        and lower bound values are the possible hkls for that peak. The index of the Q value is the index of the
        group of possible hkls. The index is recorded in the dataframe for both upper and lower bound.
        """
        dtt = self._config.index_tth_tolerance
        candidates = []
        for p in peaks:
            tt = self._peaks.loc[p, "tth"]
            left = np.searchsorted(self._pred.tth, tt - dtt, side="left")
            right = np.searchsorted(self._pred.tth, tt + dtt, side="right")
            chosen = self._pred.hkls[left:right]
            if not chosen:
                raise PeakIndexer(
                    "No candidate hkls found. Please increase the two theta tolerance."
                )
            hkls = np.concatenate(chosen)
            candidates.append(hkls)
        return candidates

    def _get_anlge_h1_h2(
        self, peak1: int, hkls1: np.ndarray, peak2: int, hkls2: np.ndarray
    ) -> typing.Generator[AngleComparsion, None, None]:
        self._set_us_for_peaks(peak1, peak2)
        angle_in_sample = self._get_angle_in_sample_frame()
        if not (1e-8 < abs(angle_in_sample) < (180.0 - 1e-8)):
            return
        n1 = hkls1.shape[0]
        n2 = hkls2.shape[0]

        def gen_pairs():
            for i in range(n1):
                for j in range(n2):
                    yield i, j
            return

        pairs = tqdm(
            gen_pairs(), total=(n1 * n2), disable=(not self._config.enable_tqdm)
        )
        for i, j in pairs:
            self._ubmatrix.set_h1_from_hkl(hkls1[i])
            self._ubmatrix.set_h2_from_hkl(hkls2[j])
            angle_in_grain = self._get_anlge_in_grain_frame()
            if 1e-8 < abs(angle_in_grain) < (180.0 - 1e-8):
                diff = abs(angle_in_grain - angle_in_sample)
                yield AngleComparsion(
                    self._ubmatrix.h1,
                    self._ubmatrix.h2,
                    angle_in_sample,
                    angle_in_grain,
                    diff,
                )
        return

    def _set_us_for_peaks(self, peak1: int, peak2: int) -> None:
        row1 = self._peaks.loc[peak1]
        row2 = self._peaks.loc[peak2]
        xy1 = np.array([row1["x"], row1["y"]])
        xy2 = np.array([row2["x"], row2["y"]])
        self._ubmatrix.set_u1_from_xy(xy1)
        self._ubmatrix.set_u2_from_xy(xy2)
        return

    def _get_angle_in_sample_frame(self) -> float:
        return _get_anlge(self._ubmatrix.u1, self._ubmatrix.u2)

    def _get_anlge_in_grain_frame(self) -> float:
        return _get_anlge(self._ubmatrix.h1, self._ubmatrix.h2)

    def _get_U(self, h1: HKL, h2: HKL) -> Matrix:
        self._ubmatrix.h1 = h1
        self._ubmatrix.h2 = h2
        self._ubmatrix.get_U()
        return self._ubmatrix.U

    def _get_losses(self, hkls: np.ndarray) -> np.ndarray:
        rhkls = np.around(hkls)
        vs = self._ubmatrix.reci_to_cart(hkls)
        rvs = self._ubmatrix.reci_to_cart(rhkls)
        diffs_sq = np.sum((rvs - vs) ** 2, axis=1)
        lens_sq = np.sum(vs**2, axis=1)
        cost = np.sqrt(diffs_sq / lens_sq)
        return cost

    def _get_indexing_result_for_peaks(self, peaks: typing.List[int]) -> np.ndarray:
        return np.stack([self._get_hkl_for_a_peak(peak) for peak in peaks])

    def _get_hkl_for_a_peak(self, peak: int) -> HKL:
        row = self._peaks.loc[peak]
        xy = np.array([row["x"], row["y"]])
        u = self._ubmatrix.xy_to_lab(xy)
        v = self._ubmatrix.lab_to_cart(u)
        hkl = self._ubmatrix.cart_to_reci(v)
        return hkl

    def show(self, best_n: T.Optional[int] = None) -> None:
        """Print out the indexing results.

        Parameters
        ----------
        best_n : T.Optional[int], optional
            Only print out best n results, by default None
        """
        if not best_n:
            best_n = self._peak_index.sizes["candidate"]
        data = self._peak_index.sortby("loss").isel({"candidate": slice(0, best_n)})
        return self._print_grouped_data(data)

    def _print_grouped_data(self, data: xr.Dataset) -> None:
        n = data.sizes["candidate"]
        for i in range(n):
            sel = data.isel({"candidate": i})
            self._print_data(sel)
        return

    def _print_data(self, data: xr.Dataset) -> None:
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
