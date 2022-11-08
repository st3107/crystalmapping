from pathlib import Path

import matplotlib.pyplot as plt

from crystalmapping.crystalmapper import CrystalMapper, MapperConfig
from crystalmapping.simulations import FourImageRunV2


def test_crystal_mapping(tmp_path: Path):
    run = FourImageRunV2()
    tmp_nc = tmp_path.joinpath("result.nc")
    config = MapperConfig(
        image_data_key=run.image_key,
        RoI_number=4,
        RoI_half_width=2,
        trackpy_kernel_size=2,
        enable_tqdm=False,
    )
    c = CrystalMapper(config)
    c.load_bluesky_v2(run)
    c.show_frame(0)
    c.find_bragg_spots()
    c.tune_RoI(4, 2)
    c.create_crystal_maps()
    c.visualize()
    c.save_dataset(str(tmp_nc))
    c = CrystalMapper(config)
    c.load_dataset(str(tmp_nc))
    c.show_windows()
    c.show_intensity()
    plt.close("all")
    return
