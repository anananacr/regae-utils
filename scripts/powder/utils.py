from typing import List, Optional, Callable, Tuple, Any, Dict
from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import sys

sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
import geometry_funcs as gf


def gaussian_lin(
    x: np.ndarray, a: float, x0: float, sigma: float, m: float, n: float
) -> np.ndarray:
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
    return m * x + n + a * exp(-((x - x0) ** 2) / (2 * sigma**2))


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


def open_fwhm_map_global_min(
    lines: list, output_folder: str, label: str, pixel_step: int
):
    """
    Open FWHM grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"], dtype=np.float64).reshape((n, n))
    r = np.array(merged_dict["r_squared"]).reshape((n, n))

    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 5
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM")

    pos2 = ax2.imshow(r, cmap="rainbow")
    step = 5
    n = z.shape[0]
    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax2.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax2.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("R²")

    proj_x = np.sum(z, axis=0)
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    index_x = np.unravel_index(np.argmin(proj_x, axis=None), proj_x.shape)
    # print(index_x)
    xc = x[index_x]
    ax3.scatter(x, proj_x, color="b")
    ax3.scatter(xc, proj_x[index_x], color="r", label=f"xc: {xc}")
    ax3.set_ylabel("Average FWHM")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("FWHM projection in x")
    ax3.legend()

    proj_y = np.sum(z, axis=1)
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    index_y = np.unravel_index(np.argmin(proj_y, axis=None), proj_y.shape)
    yc = x[index_y]
    ax4.scatter(x, proj_y, color="b")
    ax4.scatter(yc, proj_y[index_y], color="r", label=f"yc: {yc}")
    ax4.set_ylabel("Average FWHM")
    ax4.set_xlabel("yc [px]")
    ax4.set_title("FWHM projection in y")
    ax4.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/plots/fwhm_map/{label}.png")
    plt.close()
    return xc, yc

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



def open_fwhm_map(lines: list, output_folder: str, label: str, pixel_step: int):
    """
    Open FWHM grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"]).reshape((n, n))
    r = np.array(merged_dict["r_squared"]).reshape((n, n))

    index_y, index_x = np.where(z == np.min(z))
    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 5
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM")

    pos2 = ax2.imshow(r, cmap="rainbow")
    step = 5
    n = z.shape[0]
    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax2.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax2.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("R²")

    proj_x = np.sum(z, axis=0)
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)

    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(x[0], x[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax3.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Xc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax3.scatter(x, proj_x, color="b")
    ax3.set_ylabel("Average FWHM")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("FWHM projection in x")
    ax3.legend()
    xc = round((-1 * popt[1]) / (2 * popt[0]))
    # print(f"xc {round((-1*popt[1])/(2*popt[0]))}")

    proj_y = np.sum(z, axis=1)
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(y[0], y[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax4.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Yc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax4.scatter(x, proj_y, color="b")
    ax4.set_ylabel("Average FWHM")
    ax4.set_xlabel("yc [px]")
    ax4.set_title("FWHM projection in y")
    ax4.legend()
    yc = round((-1 * popt[1]) / (2 * popt[0]))
    # print(f"yc {round((-1*popt[1])/(2*popt[0]))}")

    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/plots/fwhm_map/{label}.png")
    plt.close()
    if r_squared > 0.7:
        return xc, yc
    else:
        return False
