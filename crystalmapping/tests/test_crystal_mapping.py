import matplotlib.pyplot as plt
from crystalmapping import CrystalMapper, CrystalMapperConfig
from crystalmapping.datafiles import NI_CIF_FILE, NI_PONI_FILE
from crystalmapping.simulations import FourImageRunV2


def test_crystal_mapping():
    run = FourImageRunV2()
    config = CrystalMapperConfig(
        image_data_key=run.image_key,
        RoI_number=4,
        RoI_half_width=2,
        trackpy_kernel_size=2,
    )
    c = CrystalMapper(config)
    c.load_bluesky_v2(run)
    c.load(geometry=str(NI_PONI_FILE), structure=str(NI_CIF_FILE))
    c.auto_process()
    c.show_frame(0)
    c.show_windows()
    c.show_intensity()
    c.show_crystal_maps()
    plt.close("all")
    return
