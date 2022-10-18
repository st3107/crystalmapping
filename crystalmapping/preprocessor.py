from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import xarray as xr
from xarray import Dataset

EulerAngle = Tuple[float, float, float]


def _reindex_peak_id(data_lst: List[Dataset], index: str) -> None:
    for i, data in enumerate(data_lst):
        n = data[index].shape[0]
        data[index] = [f"{j+1}_{i+1}" for j in range(n)]
    return None


def _tag_data(
    data_lst: List[Dataset], euler_angle_lst: List[EulerAngle], index: str
) -> None:
    for data, (alpha, beta, gamma) in zip(data_lst, euler_angle_lst):
        n = data[index].shape[0]
        data["alpha"] = ([index], [alpha] * n)
        data["beta"] = ([index], [beta] * n)
        data["gamma"] = ([index], [gamma] * n)
    return


def _merge_data(data_lst: List[Dataset], variables: List[str], index: str) -> Dataset:
    variables += ["alpha", "beta", "gamma"]
    res: xr.Dataset = xr.concat((d[variables] for d in data_lst), index)
    return res


def _load_data(nc_file: str) -> Dataset:
    return xr.load_dataset(nc_file)


@dataclass
class Preprocessor:
    index_name: str
    merge_vars: List[str]
    data_lst: List[Dataset] = field(default_factory=list)
    euler_angle_lst: List[EulerAngle] = field(default_factory=list)

    def load_data(
        self, data_file: str, alpha: float, beta: float, gamma: float, unit: str = "deg"
    ) -> None:
        if unit == "deg":
            alpha, beta, gamma = np.deg2rad(np.array([alpha, beta, gamma]))
        data = _load_data(data_file)
        self.data_lst.append(data)
        self.euler_angle_lst.append((alpha, beta, gamma))
        return

    def process(self) -> Dataset:
        _reindex_peak_id(self.data_lst, self.index_name)
        _tag_data(self.data_lst, self.euler_angle_lst, self.index_name)
        return _merge_data(self.data_lst, self.merge_vars, self.index_name)

    def clear(self) -> None:
        self.data_lst.clear()
        self.euler_angle_lst.clear()
        return

    def pop(self) -> None:
        self.data_lst.pop()
        self.euler_angle_lst.pop()
        return
