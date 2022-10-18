from functools import lru_cache
from typing import Tuple

import numpy as np
from diffpy.structure import Lattice, Structure, loadStructure
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

Matrix = np.ndarray
EulerAngle = Tuple[float, float, float]


class UBMatrixError(Exception):
    pass


def _cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.cross(v1, v2)


def _gram_schmidt(vs: np.ndarray) -> np.ndarray:
    us = []
    # iterate column vectors
    for v in vs.transpose():
        n = len(us)
        u = v
        for i in range(n):
            u -= _project(us[i], v)
        norm_u = np.linalg.norm(u)
        if norm_u == 0.:
            raise UBMatrixError("Norm of the vector is zero.")
        u /= norm_u
        us.append(u)
    return np.column_stack(us)


def _project(u, v) -> np.ndarray:
    """Project v on the unit vector u."""
    return np.inner(u, v) * u


def _cos(deg: float) -> float:
    return np.cos(np.deg2rad(deg))


def _sin(deg: float) -> float:
    return np.sin(np.deg2rad(deg))


def _get_U_from_cart_and_inst(
    h1: np.ndarray, h2: np.ndarray, u1: np.ndarray, u2: np.ndarray
) -> np.ndarray:
    """Calculate the U matrix from the h vectors in crystal cartesian coordinates and instrument coordinate."""
    # get the third vector perpendicular to the plane
    h3 = _cross_product(h1, h2)
    u3 = _cross_product(u1, u2)
    # gram schmidt process (with normalization)
    Tc = np.column_stack([h1, h2, h3])
    Tphi = np.column_stack([u1, u2, u3])
    Tc = _gram_schmidt(Tc)
    Tphi = _gram_schmidt(Tphi)
    # multiplication
    return np.matmul(Tphi, Tc.T)


def _get_B_from_cell(lat: Lattice) -> np.ndarray:
    """Calculate the B matrix according to the unit cell."""
    return np.array(
        [
            [lat.ar, lat.br * _cos(lat.gammar), lat.cr * _cos(lat.betar)],
            [
                0.0,
                lat.br * _sin(lat.gammar),
                -lat.cr * _sin(lat.betar) * _cos(lat.alpha),
            ],
            [0.0, 0.0, 1 / lat.c],
        ]
    )


def _get_vout_from_geo(x: float, y: float, geo: AzimuthalIntegrator) -> np.ndarray:
    """Get the output beam vector. A vector form sample to the diffraction spot."""
    xyz = np.concatenate(
        geo.calc_pos_zyx(None, np.array([y]), np.array([x]))
    ).squeeze()[::-1]
    return xyz


def _get_u_from_geo(x: float, y: float, geo: AzimuthalIntegrator) -> np.ndarray:
    """Get the unit vector of the Q."""
    vout = _get_vout_from_geo(x, y, geo)
    vout /= np.linalg.norm(vout)
    vin = np.array([0.0, 0.0, 1.0])
    # wavelength unit: m, vdiff unit: A.
    vdiff = (vout - vin) / (geo.wavelength * 1e10)
    return vdiff


@lru_cache(8)
def _get_rot_matrix(alpha: float, beta: float, gamma: float) -> Matrix:
    c1, c2, c3 = np.cos(np.array([alpha, beta, gamma]))
    s1, s2, s3 = np.sin(np.array([alpha, beta, gamma]))
    mat = np.zeros((3, 3), dtype=np.float64)
    mat[0][0] = c1 * c3 - c2 * s1 * s3
    mat[0][1] = -c1 * s3 - c2 * c3 * s1
    mat[0][2] = s1 * s2
    mat[1][0] = c3 * s1 + c1 * c2 * s3
    mat[1][1] = c1 * c2 * c3 - s1 * s3
    mat[1][2] = -c1 * s2
    mat[2][0] = s2 * s3
    mat[2][1] = c3 * s2
    mat[2][2] = c2
    return mat


def _load_lat(cif_file: str) -> Lattice:
    stru: Structure = loadStructure(cif_file)
    return stru.lattice


def _sample_to_lab(R: Matrix, v_sample: np.ndarray) -> np.ndarray:
    return np.matmul(R, v_sample.T).T


def _lab_to_sample(R: Matrix, v_lab: np.ndarray) -> np.ndarray:
    return np.matmul(R.T, v_lab.T).T


class UBMatrix:
    """Calculator for the UB matrix.

    The algorithm is based on Busing, W. R. & Levy, H. A. (1967). Acta Cryst. 22, 457–464.

    Attributes
    ----------
    h1 : 1d array
        A vector in grain frame.
    h2 : 1d array
        A vector in grain frame.
    u1 : 1d array
        A vector in sample frame.
    u2 : 1d array
        A vector in sample frame.
    lat : Lattice
        A lattice containing a, b, c, alpha, beta, gamma.
    geo : AzimuthalIntegrator
        The geometry of the sample and detector in lab frame.
    U : 2d array
        The U matrix for column vectors.
    B : 2d array
        The B matrix for column vectors.
    """

    def __init__(
        self,
        h1: np.ndarray = None,
        h2: np.ndarray = None,
        u1: np.ndarray = None,
        u2: np.ndarray = None,
        lat: Lattice = None,
        geo: AzimuthalIntegrator = None,
    ):
        self.h1 = h1
        self.h2 = h2
        self.u1 = u1
        self.u2 = u2
        self._lat: Lattice = lat
        self.geo = geo
        self.invB = None
        self.U = self.get_U() if self.able_to_get_U() else None
        self.B = self.get_B() if self.able_to_get_B() else None
        self.R1 = None
        self.R2 = None

    def able_to_get_U(self):
        """Return True if able to fil in self.U."""
        for attr in (self.h1, self.h2, self.u1, self.u2):
            if attr is None:
                return False
        return True

    def able_to_get_B(self):
        """Return True if able to fill in self.B."""
        return self._lat is not None

    def get_U(self) -> None:
        """Fill in the self.U attribute."""
        if not self.able_to_get_U():
            raise UBMatrixError("Not able to get U matrix. Attributes are missing.")
        self.U = _get_U_from_cart_and_inst(self.h1, self.h2, self.u1, self.u2)
        return

    def get_B(self) -> None:
        """Fill in the self.B attribute."""
        if not self.able_to_get_B():
            raise UBMatrixError("Not able to get B matrix. Attributes are missing.")
        self.B = _get_B_from_cell(self._lat)
        self.invB = np.linalg.inv(self.B)
        return

    def set_u1_from_xy(self, xy: np.ndarray) -> None:
        """Set self.u1 by x y coordinates using self.geo."""
        self.u1 = self.lab_to_sample_1(self.xy_to_lab(xy))
        return

    def set_u2_from_xy(self, xy: np.ndarray) -> None:
        """Set self.u2 by x y coordinate using self.geo."""
        self.u2 = self.lab_to_sample_2(self.xy_to_lab(xy))
        return

    def set_h1_from_hkl(self, hkl: np.ndarray) -> None:
        """Set self.h1 by hkl using B matrix."""
        self.h1 = self.lat_to_grain(hkl)
        return

    def set_h2_from_hkl(self, hkl: np.ndarray) -> None:
        """Set self.h2 by hkl using B matrix."""
        self.h2 = self.lat_to_grain(hkl)
        return

    def set_R1(self, alpha: float, beta: float, gamma: float) -> None:
        self.R1 = _get_rot_matrix(alpha, beta, gamma)
        return

    def set_R2(self, alpha: float, beta: float, gamma: float) -> None:
        self.R2 = _get_rot_matrix(alpha, beta, gamma)
        return

    @property
    def lat(self) -> Lattice:
        return self._lat

    @lat.setter
    def lat(self, lat: Lattice):
        self._lat = lat
        self.get_B()
        return

    def set_lat_from_cif(self, cif_file: str) -> None:
        """Set self.lat by cif file."""
        self.lat = _load_lat(cif_file)
        return

    def xy_to_lab(self, xy: np.ndarray) -> np.ndarray:
        """Transform x, y pixel coordinate to a unit vector in lab frame."""
        if self.geo is None:
            raise UBMatrixError("`self.geo` is None.")
        return _get_u_from_geo(xy[0], xy[1], self.geo)

    def sample_to_lab_1(self, v_sample: np.ndarray) -> np.ndarray:
        if self.R1 is None:
            raise UBMatrixError("self.R1 is None.")
        return _sample_to_lab(self.R1, v_sample)

    def lab_to_sample_1(self, v_lab: np.ndarray) -> np.ndarray:
        if self.R1 is None:
            raise UBMatrixError("self.R1 is None.")
        return _lab_to_sample(self.R1, v_lab)

    def sample_to_lab_2(self, v_sample: np.ndarray) -> np.ndarray:
        if self.R2 is None:
            raise UBMatrixError("self.R2 is None.")
        return _sample_to_lab(self.R2, v_sample)

    def lab_to_sample_2(self, v_lab: np.ndarray) -> np.ndarray:
        if self.R2 is None:
            raise UBMatrixError("self.R2 is None.")
        return _lab_to_sample(self.R2, v_lab)

    def lat_to_grain(self, v_lat: np.ndarray) -> np.ndarray:
        """Transform a vector from reciprocal space (hkl) frame to crystal cartesian frame."""
        if self.B is None:
            raise UBMatrixError("`self.B` is None.")
        return np.matmul(self.B, v_lat.T).T

    def grain_to_lat(self, v_grain: np.ndarray) -> np.ndarray:
        """Transform a vector from the cartesian crystal frame to reciprocal space (hkl)."""
        if self.invB is None:
            raise UBMatrixError("`self.B` is None.")
        return np.matmul(self.invB, v_grain.T).T

    def grain_to_sample(self, v_grain: np.ndarray) -> np.ndarray:
        """Transform a vector from cartesian crystal frame to lab frame."""
        if self.U is None:
            raise UBMatrixError("`self.U` is None")
        return np.matmul(self.U, v_grain.T).T

    def sample_to_grain(self, v_sample: np.ndarray) -> np.ndarray:
        """Transform a vector from the lab frame to cartesian crystal frame."""
        if self.U is None:
            raise UBMatrixError("`self.U` is None.")
        # U^T is the U^{-1}
        return np.matmul(self.U.T, v_sample.T).T

    def get_U_from_two_points(
        self, xy1: np.ndarray, hkl1: np.ndarray, xy2: np.ndarray, hkl2: np.ndarray
    ) -> None:
        """Get the U matrix from two x, y pixel coordinates on the detector and their hkl."""
        self.set_u1_from_xy(xy1)
        self.set_u2_from_xy(xy2)
        self.set_h1_from_hkl(hkl1)
        self.set_h2_from_hkl(hkl2)
        self.get_U()
        return

    def set_geo_from_poni(self, poni_file: str) -> None:
        """Set the geometry by poni file."""
        self.geo = AzimuthalIntegrator()
        self.geo.load(poni_file)
        return
