import numpy as np
from diffpy.structure import Lattice
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import crystalmapping.ubmatrix as ubmatrix


def test_UBMatrix_get_B():
    ub = ubmatrix.UBMatrix()
    ub.lat = Lattice(1., 1., 1., 90., 90., 90.)
    expect_B = np.diag([1., 1., 1.])
    assert np.allclose(ub.B, expect_B)


def test_UBMatrix_get_B_2():
    ub = ubmatrix.UBMatrix()
    ub.lat = Lattice(3., 1., 2., 90., 60., 60.)
    for i in range(3):
        for j in range(i + 1, 3):
            inner = np.dot(ub.B[i], ub.B[j])
            assert inner < 1e-8


def test_UBMatrix_get_U():
    ub = ubmatrix.UBMatrix()
    ub.geo = AzimuthalIntegrator(dist=1., pixel1=1., pixel2=1.)
    ub.u1 = np.array([1., 0., 1.])
    ub.u2 = np.array([0., 1., 1.])
    ub.h1 = np.array([1., 0., 1.])
    ub.h2 = np.array([0., 1., 1.])
    ub.get_U()
    expect_U = np.diag([1., 1., 1.])
    assert np.allclose(ub.U, expect_U)
