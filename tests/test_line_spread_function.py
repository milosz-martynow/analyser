import numpy as np
import pytest

from analyser.line_spread_function import lsf


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
def test_lsf(image_fixture_name, request):
    """
    Shared test for ``lsf`` applied to all synthetic images provided
    by the test fixtures.

    image_fixture_name : str
        Name of the pytest fixture returning a synthetic test image.

    This test verifies for each image:

    - ``lsf`` returns non-empty derivative/distance lists.
    - The number of derivative curves equals the number of distance arrays.
    - Each returned distance array is strictly increasing.
    - Each derivative has the same length as its corresponding distance array.
    """
    image = request.getfixturevalue(image_fixture_name)

    derivatives, distances = lsf(image)

    # Basic structure expectations
    assert len(derivatives) > 0
    assert len(distances) > 0
    assert len(derivatives) == len(distances)

    for d, dist in zip(derivatives, distances):
        diffs = np.diff(dist)
        assert np.all(diffs > 0)
        assert len(d) == len(dist)
