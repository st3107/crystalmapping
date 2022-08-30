import matplotlib.pyplot as plt
from crystalmapping import CrystalMapper, CrystalMapperConfig
from crystalmapping.datafiles import CEO2_PONI_FILE, CRYSTAL_MAPS_FILE, TIO2_CIF_FILE


def test_indexing_real_data():
    GRPOUP1 = [16, 59, 37]
    config = CrystalMapperConfig()
    cm = CrystalMapper(config)
    cm.load(
        geometry=str(CEO2_PONI_FILE),
        structure=str(TIO2_CIF_FILE),
        crystal_maps=str(CRYSTAL_MAPS_FILE),
    )
    cm.prepare_for_indexing()
    cm.show_crystal_maps(GRPOUP1)
    plt.close("all")
    cm.guess_miller_index(GRPOUP1)
    cm.print_indexing_result()
    return
