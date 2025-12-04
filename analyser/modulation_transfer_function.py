"""Scripts to calculate Modulation Transfer Function (MTF)
Line Spread Function (LSF) extracted from the image matrix."""

from typing import List, Tuple, Union

import numpy as np


def mtf(
    distances: Union[List[Union[int, float]], np.ndarray],
    derivative: Union[List[Union[int, float]], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Modulation Transfer Function (MTF) from a discrete
    line-spread derivative sampled at known pixel-distance locations.

    The function estimates the sampling step from the distance
    differences, computes the Fourier transform of the derivative
    (LSF), and normalizes the MTF so that MTF(0) = 1.

    :param distances:
        Monotonically increasing sampling positions along the diameter line.
    :type distances: Union[List[Union[int, float]], np.ndarray]

    :param derivative:
        Derivative (LSF) values corresponding to the sampling positions.
    :type derivative: Union[List[Union[int, float]], np.ndarray]

    :return:
        A tuple ``(frequencies, mtf)``, where

        - ``frequencies`` is an array of spatial frequencies (cycles/pixel)
        - ``mtf`` is the normalized magnitude of the MTF
    :rtype: tuple[numpy.ndarray, numpy.ndarray]

    :raises ValueError:
        If fewer than 3 samples are provided or distances are not strictly increasing.
    """
    if len(distances) < 3 or len(derivative) < 3:
        raise ValueError("At least 3 samples are required to compute MTF.")

    dist = np.asarray(distances, dtype=float)
    lsf = np.asarray(derivative, dtype=float)

    diffs = np.diff(dist)
    if not np.all(diffs > 0):
        raise ValueError(
            "Distances must be strictly increasing for valid sampling."
        )

    dx = np.mean(diffs)

    mtf_raw = np.abs(np.fft.rfft(lsf))
    frequencies = np.fft.rfftfreq(len(lsf), d=dx)

    return frequencies, mtf_raw


def mtf_nyquist_values(
    frequencies: Union[List[Union[int, float]], np.ndarray],
    mtf: Union[List[Union[int, float]], np.ndarray],
) -> Tuple[float, float]:
    """
    Compute the MTF value at the Nyquist frequency and at half the Nyquist
    frequency. The Nyquist frequency is inferred from the highest frequency
    value in the provided array.

    :param frequencies: Frequency values corresponding to the MTF curve.
    :type frequencies: Union[List[Union[int, float]], np.ndarray]
    :param mtf: MTF values associated with the frequencies.
    :type mtf: Union[List[Union[int, float]], np.ndarray]

    :return: Tuple of (MTF at Nyquist frequency, MTF at half-Nyquist frequency).
    :rtype: tuple[float, float]

    :raises ValueError: If fewer than two samples are provided.
    """
    if len(frequencies) < 2 or len(mtf) < 2:
        raise ValueError(
            "At least 2 frequency and MTF samples are required to evaluate Nyquist MTF."
        )

    frequencies = np.asarray(frequencies, dtype=float)
    mtf = np.asarray(mtf, dtype=float)

    f_nyquist = frequencies[-1]
    f_half_nyquist = f_nyquist * 0.5

    mtf_nq = np.interp(f_nyquist, frequencies, mtf)
    mtf_half_nq = np.interp(f_half_nyquist, frequencies, mtf)

    return float(mtf_nq), float(mtf_half_nq)


def mtf_fwhm(freq: np.ndarray, mtf: np.ndarray) -> float:
    """
    Compute the full width at half maximum (FWHM) of a modulation transfer
    function (MTF) curve.

    The function normalizes the MTF to its maximum value, identifies the two
    points where the MTF crosses half of this peak using linear interpolation,
    and returns the distance between them in frequency units.

    :param numpy.ndarray freq:
        1D array of monotonically increasing frequency samples.
    :param numpy.ndarray mtf:
        1D array of MTF values corresponding to ``freq``. These values do not
        need to be normalized.

    :returns:
        The FWHM of the MTF curve in frequency units (right_cross - left_cross).
        Returns ``numpy.nan`` if fewer than two half-maximum crossings are found
        or if inputs are invalid.
    :rtype: float or numpy.nan

    :raises ValueError:
        If ``freq`` and ``mtf`` do not have the same shape.
    """
    freq = np.asarray(freq, dtype=float)
    mtf_vals = np.asarray(mtf, dtype=float)

    if freq.shape != mtf_vals.shape:
        raise ValueError("freq and mtf_vals must have the same shape.")

    if len(freq) < 2:
        return float(np.nan)

    # Find half maximum
    mtf_max = np.nanmax(mtf_vals)
    if np.isnan(mtf_max) or mtf_max == 0:
        return float(np.nan)

    half_max = mtf_max / 2.0

    # Boolean array: True where mtf >= half_max
    above = mtf_vals >= half_max

    # Find indices where boolean changes (crossings between i and i+1)
    change_indices = np.where(np.diff(above.astype(int)) != 0)[0]

    # Need at least two crossings (entering and leaving the >= half region)
    if change_indices.size < 2:
        return float(np.nan)

    # Left crossing (first change) and right crossing (last change)
    left_idx = change_indices[0]
    right_idx = change_indices[-1]

    def interp_cross(i):
        """Linear interpolation of frequency where mtf == half_max between i and i+1."""
        x0, x1 = freq[i], freq[i + 1]
        y0, y1 = mtf_vals[i], mtf_vals[i + 1]

        # Avoid division by zero (flat segment)
        if y1 == y0:
            # If flat and equals half_max, return midpoint; otherwise return nearest x
            if y0 == half_max:
                return float(0.5 * (x0 + x1))
            return float(x0)

        return float(x0 + (half_max - y0) * (x1 - x0) / (y1 - y0))

    left_cross = interp_cross(left_idx)
    right_cross = interp_cross(right_idx)

    # Final FWHM
    return float(right_cross - left_cross)


def grd_from_mtf(
    freq: np.ndarray, mtf: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Compute the Ground Resolved Distance (GRD) from a Modulation Transfer
    Function (MTF) curve.

    GRD is defined as the reciprocal of the spatial frequency at which the MTF
    falls to a specified contrast threshold (e.g. 0.1 for MTF10, 0.5 for MTF50).

    Mathematically::

        GRD = 1 / f_threshold

    where ``f_threshold`` is found by locating the frequency at which
    ``MTF(f) == threshold`` using linear interpolation.

    :param numpy.ndarray freq:
        1D array of spatial frequencies (e.g. cycles/pixel or cycles/mm),
        monotonically increasing.
    :param numpy.ndarray mtf:
        1D array of MTF values corresponding to ``freq``.
        Does not need to be normalized.
    :param float threshold:
        The MTF contrast threshold used to define resolvable detail.
        Typical values:
            - ``0.1`` → MTF10 (common for imaging systems)
            - ``0.5`` → MTF50
            - mission-specific thresholds

    :returns:
        Ground resolved distance in the same spatial units as ``1/freq``.
        Returns ``numpy.nan`` if the threshold is not crossed.
    :rtype: float

    :raises ValueError:
        If ``freq`` and ``mtf`` do not have the same shape.

    .. note::
       !!! This function assumes an MTF with a single dominant peak and monotonic
       decay, as is typical of real imaging systems.

    """
    freq = np.asarray(freq, dtype=float)
    mtf_vals = np.asarray(mtf, dtype=float)

    if freq.shape != mtf_vals.shape:
        raise ValueError("freq and mtf must have the same shape.")

    if len(freq) < 2:
        return float(np.nan)

    # Create boolean mask for being at-or-above the threshold
    above = mtf_vals >= threshold

    # Find falling edges: True -> False transitions between i and i+1
    falling_idxs = np.where((above[:-1] == True) & (above[1:] == False))[0]

    # If no falling edge is found, return nan (threshold never crossed downward)
    if falling_idxs.size == 0:
        return float(np.nan)

    # Use the first falling edge (lowest frequency where it falls below threshold)
    i = int(falling_idxs[0])

    x0, x1 = freq[i], freq[i + 1]
    y0, y1 = mtf_vals[i], mtf_vals[i + 1]

    # If the interval is flat
    if y1 == y0:
        # If flat equals threshold, take midpoint. Otherwise, cannot interpolate meaningfully.
        if y0 == threshold:
            f_threshold = 0.5 * (x0 + x1)
        else:
            # fallback to the higher-frequency sample as approximate crossing
            f_threshold = float(x1)
    else:
        # linear interpolation for frequency where mtf == threshold
        f_threshold = float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    # sanity checks
    if not np.isfinite(f_threshold) or f_threshold <= 0:
        return float(np.nan)

    return 1.0 / f_threshold


def mtf_analysis(
    lsf_distances: List[np.ndarray], lsf_derivatives: List[np.ndarray]
):
    """
    Perform a batch Modulation Transfer Function (MTF) analysis from a set of
    line-spread function (LSF) distance arrays and their corresponding
    derivatives.

    This function computes several key MTF-based image quality metrics for each
    provided LSF, including:

    - MTF at the Nyquist frequency
    - MTF at half the Nyquist frequency
    - MTF full width at half maximum (FWHM)
    - Ground Resolved Distance (GRD)

    These metrics are averaged across all valid LSFs, and their means and
    standard deviations are printed.

    :param list[numpy.ndarray] lsf_distances:
        List of 1D arrays containing monotonically increasing sampling
        positions (pixel distances) along each LSF.
    :param list[numpy.ndarray] lsf_derivatives:
        List of 1D arrays containing the corresponding LSF derivative values
        (i.e., line spread function values).

    :returns:
        ``None``. The function prints statistical summaries of:

        - Mean and standard deviation of MTF at Nyquist frequency
        - Mean and standard deviation of MTF at half Nyquist
        - Mean and standard deviation of MTF FWHM
        - Mean and standard deviation of GRD

    :rtype: None

    :raises ValueError:
        If any of the individual MTF computations encounter invalid input
        (e.g., non-strictly increasing distances), the underlying functions
        may raise exceptions.

    .. note::
       The function does not return numerical arrays. If you need the computed
       MTF metrics instead of printed statistics, the implementation can be
       modified to return them.

    """

    mtf_at_nq_values = []
    mtf_at_half_nq_values = []
    mtf_fwhm_values = []
    mtf_grd_values = []

    for distances, derivatives in zip(lsf_distances, lsf_derivatives):

        mtf_frequencies, mtf_values = mtf(
            distances=distances, derivative=derivatives
        )
        mtf_at_nq, mtf_at_half_nq = mtf_nyquist_values(
            frequencies=mtf_frequencies, mtf=mtf_values
        )

        mtf_at_nq_values.append(mtf_at_nq)
        mtf_at_half_nq_values.append(mtf_at_half_nq)

        fwhm = mtf_fwhm(freq=mtf_frequencies, mtf=mtf_values)
        mtf_fwhm_values.append(fwhm)

        grd = grd_from_mtf(freq=mtf_frequencies, mtf=mtf_values)
        mtf_grd_values.append(grd)

    print(
        f"MTF @ NQ: mean {np.mean(mtf_at_nq_values)}, std {np.std(mtf_at_nq_values)}"
    )
    print(
        f"MTF @ NQ/2: mean {np.mean(mtf_at_half_nq_values)}, std {np.std(mtf_at_half_nq_values)}"
    )
    print(
        f"FWHM of MTF: mean {np.nanmean(mtf_fwhm_values)}, std {np.nanstd(mtf_fwhm_values)}"
    )
    print(
        f"GRD: mean {np.nanmean(mtf_grd_values)}, std {np.nanstd(mtf_grd_values)}"
    )
