import numpy as np
from diffpy.structure import Lattice
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

import crystalmapping.ubmatrix as ubmatrix


def test_UBMatrix_get_B():
    ub = ubmatrix.UBMatrix()
    ub.lat = Lattice(1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    expect_B = np.diag([1.0, 1.0, 1.0])
    assert np.allclose(ub.B, expect_B)


def test_UBMatrix_get_B_2():
    ub = ubmatrix.UBMatrix()
    ub.lat = Lattice(3.0, 1.0, 2.0, 90.0, 60.0, 60.0)
    for i in range(3):
        for j in range(i + 1, 3):
            inner = np.dot(ub.B[i], ub.B[j])
            assert inner < 1e-8


def test_UBMatrix_get_U():
    ub = ubmatrix.UBMatrix()
    ub.geo = AzimuthalIntegrator(dist=1.0, pixel1=1.0, pixel2=1.0)
    ub.u1 = np.array([1.0, 0.0, 1.0])
    ub.u2 = np.array([0.0, 1.0, 1.0])
    ub.h1 = np.array([1.0, 0.0, 1.0])
    ub.h2 = np.array([0.0, 1.0, 1.0])
    ub.calc_and_set_U()
    expect_U = np.diag([1.0, 1.0, 1.0])
    assert np.allclose(ub.U, expect_U)


def test_euler_u_mat_conversion():
    euler_angle = tuple(np.deg2rad([30.0, 60.0, 45.0]))
    u_mat = ubmatrix._euler_to_u_mat(euler_angle)
    euler_angle2 = ubmatrix._u_mat_to_euler(u_mat)
    assert np.allclose(euler_angle, euler_angle2, atol=1e-8)
    return
