"""Fresnel reflection at a single interface — the numpy reference implementation.

Spec §4.2.
Implemented by DTFM-006; ported to torch by DTFM-008.

This module is the ground truth the torch port is checked against, so it stays
plain numpy and deliberately simple. Real refractive indices only: complex `ñ`
for absorbing media, and the branch-cut selection it requires, belong to
DTFM-010 and to `tmm_torch`.

Sign convention
---------------
Spec §4.2 defines, for an interface between incident medium `i` and transmitted
medium `j`::

    r_s = (n_i cosθ_i − n_j cosθ_j) / (n_i cosθ_i + n_j cosθ_j)
    r_p = (n_j cosθ_i − n_i cosθ_j) / (n_j cosθ_i + n_i cosθ_j)

At normal incidence these differ by a sign: ``r_s = (n_i − n_j)/(n_i + n_j)``
while ``r_p = −(n_i − n_j)/(n_i + n_j)``. Both give the same reflectance, since
``R = |r|²``. That is a property of the convention, not a bug — the p-axis is
defined relative to the propagation direction, which reverses on reflection, so
at normal incidence "s" and "p" describe the same physical field with opposite
sign attached. Other texts flip `r_p`; mixing conventions between this module
and the transfer matrices in §4.3 would put a spurious sign in every stack, so
the §4.2 form is used everywhere and asserted in the tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["cos_theta_t", "fresnel_r", "reflectance"]


def cos_theta_t(n_i: ArrayLike, n_j: ArrayLike, theta_i: ArrayLike) -> NDArray[np.complexfloating]:
    """Cosine of the transmitted angle, from Snell's law ``n_i sinθ_i = n_j sinθ_j``.

    Returns a complex array. Beyond the critical angle — possible whenever
    ``n_i > n_j`` — ``1 − sin²θ_j`` is negative and the cosine is imaginary,
    describing an evanescent wave with ``|r| = 1``. Using ``numpy.emath.sqrt``
    rather than ``numpy.sqrt`` keeps that case exact instead of returning a
    silent ``nan`` in the middle of an array.

    Parameters
    ----------
    n_i, n_j : refractive index of the incident and transmitted media.
    theta_i : angle of incidence, in radians, measured from the surface normal.
    """
    n_i, n_j, theta_i = np.asarray(n_i), np.asarray(n_j), np.asarray(theta_i)
    sin_theta_t = (n_i / n_j) * np.sin(theta_i)
    return np.emath.sqrt(1.0 - sin_theta_t**2)


def fresnel_r(
    n_i: ArrayLike, n_j: ArrayLike, theta_i: ArrayLike
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Amplitude reflection coefficients ``(r_s, r_p)`` at one interface.

    Spec §4.2. Inputs broadcast against each other, so a wavelength-dependent
    index and a swept angle can be evaluated in one call.

    Returns complex arrays: the coefficients are real below the critical angle
    for real indices, but pick up a phase under total internal reflection.
    """
    n_i, n_j, theta_i = np.asarray(n_i), np.asarray(n_j), np.asarray(theta_i)
    cos_i = np.cos(theta_i).astype(complex)
    cos_j = cos_theta_t(n_i, n_j, theta_i)

    r_s = (n_i * cos_i - n_j * cos_j) / (n_i * cos_i + n_j * cos_j)
    r_p = (n_j * cos_i - n_i * cos_j) / (n_j * cos_i + n_i * cos_j)
    return r_s, r_p


def reflectance(
    n_i: ArrayLike, n_j: ArrayLike, theta_i: ArrayLike
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Reflected intensity fraction ``(R_s, R_p) = (|r_s|², |r_p|²)``."""
    r_s, r_p = fresnel_r(n_i, n_j, theta_i)
    return np.abs(r_s) ** 2, np.abs(r_p) ** 2
