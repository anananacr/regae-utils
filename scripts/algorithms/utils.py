from typing import List, Optional, Callable, Tuple, Any, Dict
from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import sys

sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
# from maskMakerGUI import pMakePolarisationArray as make_polarization_array_fast
import geometry_funcs as gf


def center_of_mass(data: np.ndarray, mask: np.ndarray = None) -> Tuple[int]:
    """
    Adapted from Robert Bücker work on diffractem (https://github.com/robertbuecker/diffractem/tree/master)
    Bücker, R., Hogan-Lamarre, P., Mehrabi, P. et al. Serial protein crystallography in an electron microscope. Nat Commun 11, 996 (2020). https://doi.org/10.1038/s41467-020-14793-0

    Parameters
    ----------
    data: np.ndarray
        Input data in which center of mass will be calculated. Values equal or less than zero will not be considered.
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns
    ----------
    xc, yc: int
         coordinates of the diffraction center in x and y, such as the image center corresponds to data[yc, xc].
    """

    if mask is None:
        mask = np.ones_like(data)
    data = data * mask
    indices = np.where(data > 0)
    if np.any(data):
        xc = np.sum(data[indices] * indices[1]) / np.sum(data[indices])
        yc = np.sum(data[indices] * indices[0]) / np.sum(data[indices])
    else:
        xc = 0
        yc = 0

    return xc, yc


def azimuthal_average(
    data: np.ndarray, center: tuple = None, mask: np.ndarray = None
) -> np.ndarray:
    """
    Calculate azimuthal integration of data in relation to the center of the image
    Adapted from L. P. René de Cotret work on scikit-ued (https://github.com/LaurentRDC/scikit-ued/tree/master)
    L. P. René de Cotret, M. R. Otto, M. J. Stern. and B. J. Siwick, An open-source software ecosystem for the interactive exploration of ultrafast electron scattering data, Advanced Structural and Chemical Imaging 4:11 (2018) DOI: 10.1186/s40679-018-0060-y.

    Parameters
    ----------
     data: np.ndarray
        Input data in which center of mass will be calculated. Values equal or less than zero will not be considered.
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.
    Returns
    ----------
    radius: np.ndarray
        radial axis radius in pixels

    intensity: np.ndarray
        Integrated intensity normalized by the number of valid pixels
    """
    a = data.shape[0]
    b = data.shape[1]
    if mask is None:
        mask = np.ones((a, b), dtype=bool)
    else:
        mask.astype(bool)
    if center is None:
        center = [b / 2, a / 2]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    Rint = np.rint(R).astype(int)

    valid = mask.flatten()
    data = data.flatten()
    Rint = Rint.flatten()

    px_bin = np.bincount(Rint, weights=valid * data)
    r_bin = np.bincount(Rint, weights=valid)
    radius = np.arange(0, r_bin.size)
    # Replace by one if r_bin is zero for division
    np.maximum(r_bin, 1, out=r_bin)

    return radius, px_bin / r_bin


def open_distance_map_global_min(
    lines: list, output_folder: str, label: str, pixel_step: int
) -> tuple:
    """
    Open distance minimization plot, fit projections in both axis to get the point of minimum distance.

    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
    """

    n = int(math.sqrt(len(lines)))
    pixel_step /= 2
    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["d"], dtype=np.float64).reshape((n, n))

    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 20
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=float))
    ax1.set_yticks(np.arange(0, n, step, dtype=float))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)

    ax1.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax1.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("Distance [px]")

    proj_x = np.sum(z, axis=0) / z.shape[0]
    # print('proj',len(proj_x))
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    # print('x',len(x))
    index_x = np.unravel_index(np.argmin(proj_x, axis=None), proj_x.shape)
    # print(index_x)
    xc = round(x[index_x], 1)
    ax2.scatter(x, proj_x + pixel_step, color="b")
    ax2.scatter(xc, proj_x[index_x], color="r", label=f"xc: {xc}")
    ax2.set_ylabel("Average distance [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("Distance projection in x")
    ax2.legend()

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    index_y = np.unravel_index(np.argmin(proj_y, axis=None), proj_y.shape)
    yc = round(x[index_y], 1)
    ax3.scatter(x, proj_y, color="b")
    ax3.scatter(yc, proj_y[index_y], color="r", label=f"yc: {yc}")
    ax3.set_ylabel("Average Distance [px]")
    ax3.set_xlabel("yc [px]")
    ax3.set_title("Distance projection in y")
    ax3.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/plots/distance_map/{label}.png")
    plt.close()
    if int(np.sum(proj_y)) == 0 or int(np.sum(proj_x)) == 0:
        converged = 0
    else:
        converged = 1
    return xc, yc, converged


def open_distance_map_fit_min(
    lines: list, output_folder: str, label: str, pixel_step: int
) -> tuple:
    """
    Open distance minimization plot, fit projections in both axis to get the point of minimum distance.

    Parameters
    ----------
    lines: list
        Output of grid search for distance optmization, each line should contain a dictionary contaning entries for xc, yc and distance.
    """

    n = int(math.sqrt(len(lines)))
    pixel_step /= 2
    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["d"], dtype=np.float64).reshape((n, n))

    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 20
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=float))
    ax1.set_yticks(np.arange(0, n, step, dtype=float))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)

    ax1.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax1.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("Distance [px]")

    proj_x = np.sum(z, axis=0) / z.shape[0]
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)

    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(x[0], x[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)

    a, b, c = popt
    xc = round((-1 * b) / (2 * a), 1)
    y_min = round((-1 * (b**2) + 4 * a * c) / (4 * a), 1)
    ax2.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, xc: {xc}",
    )

    ax2.scatter(x, proj_x + pixel_step, color="b")
    ax2.scatter(xc, y_min, color="r", label=f"xc: {xc}")
    ax2.set_ylabel("Average distance (px)")
    ax2.set_xlabel("Detector center in x (px)")
    ax2.set_title("Distance projection in x")
    ax2.legend()

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    a, b, c = popt
    yc = round((-1 * b) / (2 * a), 1)
    y_min = round((-1 * (b**2) + 4 * a * c) / (4 * a), 1)

    x_fit = np.arange(y[0], y[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax3.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, yc: {yc}",
    )

    ax3.scatter(x, proj_y, color="b")
    ax3.scatter(yc, y_min, color="r", label=f"yc: {yc}")
    ax3.set_ylabel("Average distance (px)")
    ax3.set_xlabel("Detector center in y (px)")
    ax3.set_title("Distance projection in y")
    ax3.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/plots/distance_map_fit/{label}.png")
    plt.close()
    if int(np.sum(proj_y)) == 0 or int(np.sum(proj_x)) == 0:
        converged = 0
    else:
        converged = 1
    return xc, yc, converged


def mask_peaks(mask: np.ndarray, indices: tuple, bragg: int) -> np.ndarray:
    """
    Gather coordinates of a box of 1x1 pixels around each point from the indices list. Bragg flag indicates if the mask returned will contain only bragg peaks regions (bragg =1), no bragg peaks regions (bragg=0), or both (bragg =-1).
    Parameters
    ----------
    mask: np.ndarray
        An array where mask will be built based on its shape. Mask shape is the same size of data.
    indices: tuple
        Bragg peaks coordinates, indices[0] contains x-coordinates of Bragg peaks and indices[1] the corresponding y-coordinates.
    bragg: int
        Bragg flag, choose between return only peaks, only background or both (bypass masking of peaks).
    Returns
    ----------
    surrounding_mask: np.ndarray
        Corresponding mask according to bragg flag choice. It contains zeros for unvalid pixels and one for valid pixels. Mask shape is the same size of data.
    """
    surrounding_positions = []
    count = 0
    for index in zip(indices[0], indices[1]):
        n = 4
        row, col = index
        for i in range(-n, n + 1):
            for k in range(-n, n + 1):
                surrounding_positions.append((row + i, col + k))
        count += 1

    # print(args.bragg)
    if bragg == 1:
        surrounding_mask = np.zeros_like(mask)
        for pos in surrounding_positions:
            row, col = pos
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
                surrounding_mask[row, col] = 1
    elif bragg == -1:
        surrounding_mask = np.ones_like(mask)
    else:
        surrounding_mask = np.ones_like(mask)
        for pos in surrounding_positions:
            row, col = pos
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
                surrounding_mask[row, col] = 0

    return surrounding_mask


def get_format(file_path: str) -> str:
    """
    Return file format with only alphabet letters.
    Parameters
    ----------
    file_path: str

    Returns
    ----------
    extension: str
        File format contanining only alphabetical letters
    """
    ext = (file_path.split("/")[-1]).split(".")[-1]
    filt_ext = ""
    for i in ext:
        if i.isalpha():
            filt_ext += i
    return str(filt_ext)


def gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """
    Gaussian function.

    Parameters
    ----------
    x: np.ndarray
        x array of the spectrum.
    a, x0, sigma: float
        gaussian parameters

    Returns
    ----------
    y: np.ndarray
        value of the function evaluated
    """
    return a * exp(-((x - x0) ** 2) / (2 * sigma**2))


def quadratic(x, a, b, c):
    """
    Quadratic function.

    Parameters
    ----------
    x: np.ndarray
        x array of the spectrum.
    a, b, c: float
        quadratic parameters

    Returns
    ----------
    y: np.ndarray
        value of the function evaluated
    """
    return a * x**2 + b * x + c


def open_fwhm_map(lines: list, label: str = None):
    """
    Open FWHM/R grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM/R optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
    """
    n = int(math.sqrt(len(lines)))

    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"]).reshape((n, n))

    index_y, index_x = np.where(z == np.min(z))
    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 10
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax1.set_ylabel("yc (px)")
    ax1.set_xlabel("xc (px)")
    ax1.set_title("FWHM")
    proj_x = np.sum(z, axis=0) / z.shape[0]
    x = np.arange(x[0], x[-1] + 1, 1)

    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(x[0], x[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax2.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Xc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax2.scatter(x, proj_x, color="b")
    ax2.set_ylabel("Average FWHM")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("FWHM projection in x")
    ax2.legend()
    print(f"xc {round((-1*popt[1])/(2*popt[0]))}")

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + 1, 1)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(y[0], y[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax3.plot(
        y_fit,
        x_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Yc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax3.scatter(proj_y, x, color="b")
    ax3.set_xlabel("Average FWHM/R")
    ax3.set_ylabel("yc [px]")
    ax3.set_title("FWHM projection in y")
    ax3.legend()
    print(f"yc {round((-1*popt[1])/(2*popt[0]))}")

    fig.colorbar(pos1, ax=ax1, shrink=0.6)

    # Display the figure

    plt.show()
    # plt.savefig(f'/home/rodria/Desktop/20230814/fwhm_map/lyso_{label}.png')
    # plt.close()


def fit_fwhm(lines: list) -> Tuple[int]:
    """
    Find minimum of FWHM/R grid search. Fits projections in both axis to get the point of maximum sharpness of the radial average, that will correspond to the center of diffraction.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM/R optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.

    Returns
    ----------
    xc, yc: int
        coordinates of the diffraction center in x and y, such as the image center corresponds to data[yc, xc].
    """
    n = int(math.sqrt(len(lines)))
    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm_over_radius"]).reshape((n, n))

    proj_x = np.sum(z, axis=0) / z.shape[0]
    x = np.arange(x[0], x[-1] + 1, 1)
    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared_x = 1 - (ss_res / ss_tot)
    xc = round((-1 * popt[1]) / (2 * popt[0]))

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + 1, 1)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared_y = 1 - (ss_res / ss_tot)
    yc = round((-1 * popt[1]) / (2 * popt[0]))

    return xc, yc


def shift_image_by_n_pixels(data: np.ndarray, n: int, axis: int) -> np.ndarray:
    """
    Linear translation of image by n pixels in given axis. Empty values in the shifted image is filled with zero.

    Parameters
    ----------
    data: np.ndarray
        Input image to be shifted
    n: int
        Number of pixels to be shifted.
    axis: int
        Axis in which the image will be shifted. Axis 0 corresponds to a shift in the rows (y-axis), axis 1 shifts in the columns (x-axis).
    Returns
    ----------
    shifted_data: np.ndarray
        Data shifted by n pixels in axis.
    """
    max_row, max_col = data.shape
    # print(max_row,max_col)
    if axis == 1 and n >= 0:
        shifted_data = np.pad(data, pad_width=[(0, 0), (abs(n), 0)], mode="constant")
        image_cut = shifted_data[:max_row, :max_col]
    elif axis == 1 and n < 0:
        shifted_data = np.pad(data, pad_width=[(0, 0), (0, abs(n))], mode="constant")
        image_cut = shifted_data[:max_row, abs(n) :]
    elif axis == 0 and n >= 0:
        shifted_data = np.pad(data, pad_width=[(abs(n), 0), (0, 0)], mode="constant")
        image_cut = shifted_data[:max_row, :max_col]
    elif axis == 0 and n < 0:
        shifted_data = np.pad(data, pad_width=[(0, abs(n)), (0, 0)], mode="constant")
        image_cut = shifted_data[abs(n) :, :max_col]
    # print("Image cut shape", image_cut.shape)
    return image_cut


def correct_polarization(
    x: np.ndarray, y: np.ndarray, dist: float, data: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Correct data for polarisation effect, C built function from https://github.com/galchenm/vdsCsPadMaskMaker/blob/main/SubLocalBG.c#L249
    Acknowledgements: Oleksandr Yefanov, Marina Galchenkova
    Parameters
    ----------
    x: np.ndarray
        x distance coordinates from the direct beam position.
    y: np.ndarray
        y distance coordinates from the direct beam position.
    dist: float
        z distance coordinates of the detector position.
    data: np.ndarray
        Raw data frame in which polarization correction will be applied.
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns
    ----------
    corrected_data: np.ndarray
        Corrected data frame for polarization effect.
    pol: np.ndarray
        Polarization array for polarization correction.
    """

    mask = mask.astype(bool)
    mask = ~mask.flatten()
    Int = np.reshape(data.copy(), len(mask))
    pol = mask.copy().astype(np.float32)
    pol = make_polarization_array_fast(
        pol, len(mask), x.flatten(), y.flatten(), dist / Res, 0.5
    )
    mask = ~mask
    pol[np.where(mask == 0)] = 1
    Int = Int / pol
    return Int.reshape(data.shape), pol.reshape(data.shape)


def circle_mask(data: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """
    Make a  ring mask for the data

    Parameters
    ----------
    data: np.ndarray
        Image in which mask will be shaped
    radius: int
        Outer radius of the mask

    Returns
    ----------
    mask: np.ndarray
    """

    bin_size = bin
    a = data.shape[0]
    b = data.shape[1]

    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    return np.less(R, radius)
