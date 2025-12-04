"""Scripts to extract Line Spread Function from the image matrix."""

import itertools
import math
import random
from typing import List, Tuple, Union

import numpy as np

from analyser.environment import NUMBER_OF_LSF


def _edge_pixel_pairs(rows: int, cols: int) -> Tuple:
    """
    Generate all unique pairs of edge pixels in an ``N × M`` image grid.

    Edge pixels are defined as all coordinates located on the border
    of the matrix, i.e. pixels in the first or last row or in the first
    or last column. The function creates all unique unordered pairs
    of these edge pixels:

    - each pair ``(A, B)`` appears only once,
    - the reversed pair ``(B, A)`` is not included,
    - pairs of identical pixels ``(A, A)`` are not produced.

    This is equivalent to generating all 2-element combinations
    of the set of edge coordinates.

    :param int rows:
        Number of rows in the image.
    :param int cols:
        Number of columns in the image.

    :return:
        List of unique unordered pairs of edge pixel coordinates.
        Each coordinate is stored as ``(row, col)``.
    :rtype: Tuple
    """
    edge = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1
    ]

    return [(a, b) for a, b in itertools.combinations(edge, 2)]


def _line_points(
    image: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Sample unique pixel coordinates along a continuous parametric line
    between two points in a 2D image.

    :param image: 2D grayscale image matrix.
    :type image: numpy.ndarray
    :param p1: Start point ``(row, col)``.
    :type p1: tuple[int, int]
    :param p2: End point ``(row, col)``.
    :type p2: tuple[int, int]

    :return: A tuple ``(coordinates, values)`` where:
             - ``coordinates`` is a list of unique ``(row, col)`` pixel coordinates
             - ``values`` is a list of corresponding pixel intensities
    :rtype: tuple[list[tuple[int, int]], list[int]]
    """
    x1, y1 = p1
    x2, y2 = p2

    steps = max(abs(x2 - x1), abs(y2 - y1))
    steps = max(1, steps)  # avoid division by zero for identical points

    coordinates: List[Tuple[int, int]] = []
    intensities: List[int] = []

    seen = set()

    for i in range(steps + 1):
        t = i / steps
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))

        if not (0 <= x < image.shape[0] and 0 <= y < image.shape[1]):
            # skip out-of-bounds
            continue

        if (x, y) in seen:
            # avoid duplicates
            continue

        seen.add((x, y))
        coordinates.append((x, y))
        intensities.append(image[x, y])

    return coordinates, intensities


def _relative_distances(coordinates: List[Tuple[int, int]]) -> List[float]:
    """
    Compute cumulative relative distances (in pixels) along a sequence of
    diameter coordinates, from the first point to the last.

    :param coordinates: Ordered list of ``(row, col)`` coordinates along a line.
    :type coordinates: list[tuple[int, int]]

    :return: List of distances where the first value is 0.0 and the last
             equals the total line length in pixels.
    :rtype: list[float]
    """
    if len(coordinates) < 2:
        return [0.0]

    distances = [0.0]

    for i in range(1, len(coordinates)):
        x1, y1 = coordinates[i - 1]
        x2, y2 = coordinates[i]
        d = math.hypot(x2 - x1, y2 - y1)
        distances.append(distances[-1] + d)

    return distances


def _intensity_derivative(
    distances: List[Union[int, float]],
    intensities: List[Union[int, float]],
) -> np.ndarray:
    """
    Compute the rate of change of pixel intensity with respect to distance.

    The function returns the intensity derivative evaluated at each point
    along the provided distance profile. At least three samples are required.

    :param distances: Ordered list of distance values.
    :type distances: list[Union[int, float]]
    :param intensities: Pixel intensities corresponding to the distances.
    :type intensities: list[Union[int, float]]

    :return: Derivative of intensity at each distance.
    :rtype: numpy.ndarray

    :raises ValueError: If fewer than three samples are provided.
    """
    distances = np.asarray(distances, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if len(distances) < 3:
        raise ValueError(
            f"At least 3 sample points are required to compute the derivative "
            f"(received {len(distances)})."
        )

    return np.gradient(intensities, distances)


def lsf(image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute line spread functions (LSFs) for the longest valid lines connecting
    pairs of edge pixels in a 2D grayscale image.

    The function evaluates all unique pairs of edge pixels, extracts line
    coordinates and intensities, computes relative distances, and determines
    which lines represent the longest possible diameter-like profiles. Only
    these longest lines are used to compute intensity derivatives (LSFs).

    :param image: 2D array representing a grayscale image.
    :type image: numpy.ndarray
    :return: Derivative of intensity at each distance.
    :rtype: numpy.ndarray
    :return: A tuple ``(derivatives, distances)`` where:
             - ``derivatives`` List of intensity derivative arrays (LSFs) for each longest line.
             - ``values`` List of relative distance arrays corresponding to each derivative.
    :rtype: tuple[list[numpy.ndarray], list[numpy.ndarray]]
    """
    derivatives: List[np.ndarray] = []
    distances: List[np.ndarray] = []

    image_edge_points = _edge_pixel_pairs(
        rows=image.shape[0], cols=image.shape[1]
    )
    image_edge_points[:] = [
        image_edge_points[i]
        for i in random.sample(range(0, len(image_edge_points)), NUMBER_OF_LSF)
    ]

    for edge_points in image_edge_points:
        line_coordinates, line_intensities = _line_points(
            image=image, p1=edge_points[0], p2=edge_points[1]
        )
        line_distances = _relative_distances(coordinates=line_coordinates)

        if len(line_distances) < 3:
            continue
        if line_distances[-1] < np.min(image.shape) or line_distances[
            -1
        ] > np.max(image.shape):
            continue

        derivatives.append(
            _intensity_derivative(
                distances=line_distances, intensities=line_intensities
            )
        )
        distances.append(line_distances)

    return derivatives, distances
