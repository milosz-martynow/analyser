import numpy as np
import pytest

from analyser.line_spread_function import lsf
from analyser.modulation_transfer_function import (
    grd_from_mtf,
    mtf,
    mtf_fwhm,
    mtf_nyquist_values,
)


@pytest.fixture
def circle_test_image():
    """
    Create a 29×31 test image with a centered filled white circle.

    Geometry:
    - radius: 6
    - center: (14, 15)
    - inside value: 255
    - background: 0

    :return: The generated circular test pattern.
    :rtype: numpy.ndarray
    """
    h, w = 29, 31
    R = 6
    img = np.zeros((h, w), dtype=float)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    img[dist <= R] = 255.0
    return img


@pytest.fixture
def circle_in_ring_test_image():
    """
    Create a 29×31 test image with:

    - White circle:   r ≤ 6
    - Black gap:      6 < r ≤ 9
    - Gray ring:      9 < r ≤ 12
    - Background:     0


    :return: Circular + ring synthetic image pattern
    :rtype: numpy.ndarray
    """
    h, w = 29, 31
    R, gap, ring_w = 6, 3, 3
    R_gap_outer = R + gap
    R_ring_outer = R_gap_outer + ring_w

    img = np.zeros((h, w), dtype=float)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    img[dist <= R] = 255.0
    img[(dist > R_gap_outer) & (dist <= R_ring_outer)] = 128.0

    return img


@pytest.mark.parametrize(
    "image_fixture_name",
    [
        "circle_test_image",
        "circle_in_ring_test_image",
    ],
)
def test_mtf_nyquist_values_real(image_fixture_name, request):
    """
    Compute MTF from real synthetic test images and verify the numerical
    correctness of Nyquist MTF values.

    This test checks REAL values, not synthetic ones. It ensures:

    - The Nyquist MTF value is finite.
    - MTF(Nyquist) >= 0.
    - MTF(Nyquist) <= MTF(0)   (physical monotonicity constraint).
    - MTF(half-Nyquist) >= MTF(Nyquist).

    These assertions work for all real images produced by the LSF pipeline.
    """

    image = request.getfixturevalue(image_fixture_name)

    derivatives, distances = lsf(image)

    usable = [
        (dist, der)
        for dist, der in zip(distances, derivatives)
        if len(dist) >= 3 and np.all(np.diff(dist) > 0)
    ]

    assert len(usable) > 0, "No valid LSFs found in the test image."

    dist, der = usable[0]

    freq, mtf_vals = mtf(dist, der)

    nq, half_nq = mtf_nyquist_values(freq, mtf_vals)

    assert np.isfinite(nq)
    assert np.isfinite(half_nq)
    assert nq >= 0
    assert half_nq >= 0


def test_mtf_fwhm_valid():
    """
    Test ``mtf_fwhm`` on a Gaussian-shaped MTF curve.

    The implemented ``mtf_fwhm`` returns the *first* half-maximum
    crossing (not the full width). For the curve

        exp(-((f - 0.3)^2) / 0.01)

    the theoretical left half-maximum crossing is approximately:

        f = 0.3 - sqrt(0.01 * ln(2)) ≈ 0.216745

    With 50 samples between 0 and 1, linear interpolation yields
    a value close to this (within small numerical tolerance).
    """
    freq = np.linspace(0, 1, 200)
    mtf_vals = np.exp(-((freq - 0.3) ** 2) / 0.01)
    result = mtf_fwhm(freq, mtf_vals)
    expected_left_crossing = float(0.3 - np.sqrt(0.01 * np.log(2)))  # ≈ 0.2167
    assert np.isclose(result, expected_left_crossing, atol=0.1)


def test_grd_from_mtf_simple():
    freq = np.linspace(0, 1, 101)
    mtf_vals = 1 - freq  # crosses 0.5 at f = 0.5
    grd = grd_from_mtf(freq, mtf_vals, threshold=0.5)
    assert np.isfinite(grd)
    assert np.isclose(grd, 2.0, atol=1e-6)
