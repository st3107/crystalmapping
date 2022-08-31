import matplotlib.pyplot as plt
from crystalmapping.crystalmapper import CrystalMapper, MapperConfig
from crystalmapping.simulations import FourImageRunV2


def test_crystal_mapping():
    run = FourImageRunV2()
    config = MapperConfig(
        image_data_key=run.image_key,
        RoI_number=4,
        RoI_half_width=2,
        trackpy_kernel_size=2,
        enable_tqdm=False
    )
    c = CrystalMapper(config)
    c.load_bluesky_v2(run)
    c.auto_process()
    c.show_frame(0)
    c.show_windows()
    c.show_intensity()
    c.visualize()
    plt.close("all")
    return
