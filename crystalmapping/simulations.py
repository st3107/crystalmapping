import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

from crystalmapping.datafiles import FAKE_IMAGE_FILE


class Primary:
    def __init__(self, data: Dataset) -> None:
        self._data = data

    def to_dask(self) -> Dataset:
        return self._data


class Metadata(dict):
    def __init__(self, start: dict):
        super().__init__()
        self["start"] = start


class FakeBlueskyRunV2:
    def __init__(self, data: Dataset, start: dict) -> None:
        self.primary: Primary = Primary(data)
        self.metadata: Metadata = Metadata(start)


class FakeBlueskyRunV1(FakeBlueskyRunV2):
    def start(self) -> dict:
        return self.metadata["start"]

    def xarray_dask(self) -> Dataset:
        return self.primary.to_dask()


class FourImageRunV2(FakeBlueskyRunV2):
    def __init__(self) -> None:
        light_image: np.ndarray = 1. - plt.imread(str(FAKE_IMAGE_FILE))
        light_image = np.expand_dims(light_image, 0)
        dark_image: np.ndarray = np.zeros_like(light_image)
        image_key = "dexela_image"
        data = Dataset(
            {
                image_key: (
                    ["time", "dim_0", "dim_1", "dim_2"],
                    [light_image, light_image, dark_image, dark_image],
                )
            }
        )
        start = {
            "shape": [2, 2],
            "extents": [(0, 1), (0, 1)],
            "snaking": (False, False),
        }
        super().__init__(data, start)
        self.image_key = image_key


class EightImageRunV2(FakeBlueskyRunV2):
    def __init__(self) -> None:
        light_image: np.ndarray = 1. - plt.imread(str(FAKE_IMAGE_FILE))
        light_image = np.expand_dims(light_image, 0)
        dark_image: np.ndarray = np.zeros_like(light_image)
        image_key = "dexela_image"
        data = Dataset(
            {
                image_key: (
                    ["time", "dim_0", "dim_1", "dim_2"],
                    [dark_image, light_image, dark_image, light_image, dark_image, light_image, dark_image, light_image],
                ),
                "x": (["time"], [0, 0, 0, 0, 1, 1, 1, 1]),
                "y": (["time"], [0, 0, 1, 1, 0, 0, 1, 1]),
                "phi": (['time'], [0, 1, 0, 1, 0, 1, 0, 1])
            }
        )
        start = {
            "shape": [2, 2, 2],
            "extents": [(0, 1), (0, 1), (0, 1)],
            "snaking": (False, False, False)
        }
        super().__init__(data, start)
        self.image_key = image_key
