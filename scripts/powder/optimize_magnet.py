import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as color
import imageio
import subprocess as sub
from PIL import Image
import math
import os
import cv2
from numpy import exp
from utils import azimuthal_average, gaussian
from scipy.signal import find_peaks as find_peaks
from scipy.optimize import curve_fit

center = [
    [581, 562],
    [581, 562],
    [583, 562],
    [585, 560],
    [586, 558],
    [587, 555],
    [587, 554],
    [587, 552],
    [587, 549],
    [587, 545],
    [587, 540],
]
height = [100, 100, 100, 100, 100, 100, 100, 100, 50, 5, 5]

max_frames = 11


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


def fit_gaussian(x: list, y: list, peak_position: int):

    a = y[peak_position]
    x = x[peak_position - 20 : peak_position + 20]
    y = y[peak_position - 20 : peak_position + 20]

    m0 = (y[-1] - y[0]) / (x[-1] - x[0])
    n0 = ((y[-1] + y[0]) - m0 * (x[-1] + x[0])) / 2
    y_linear = m0 * x + n0
    y_gaussian = y - y_linear

    mean = sum(x * y_gaussian) / sum(y_gaussian)
    sigma = np.sqrt(sum(y_gaussian * (x - mean) ** 2) / sum(y_gaussian))

    popt, pcov = curve_fit(
        gaussian_lin, x, y, p0=[max(y_gaussian), mean, sigma, m0, n0]
    )
    fwhm = popt[2] * math.sqrt(8 * np.log(2))
    ## Divide by radius of the peak to get shasrpness ratio
    fwhm_over_radius = fwhm / popt[1]

    ##Calculate residues
    residuals = y - gaussian_lin(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return fwhm_over_radius, popt, r_squared


def get_minimum(file_path: str, output: str):
    ## Rings radius in pixels
    rings = [10]
    ## Center of the image [xc,yc]

    frames = np.arange(0, max_frames, 1)
    fwhm_over_radius = []
    for frame in frames:

        data = np.array(Image.open(f"{file_path}_{frame:06}.tif"))
        mask = np.ones_like(data)
        mask[np.where(data == -1)] = 0
        bins, counts = azimuthal_average(data, center[frame], mask)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(bins, counts)
        ax.set_ylim(0, 500)
        ax.set_xlim(0, 350)
        ax.set_xlabel("Radial distance (pixel)")
        ax.set_ylabel("Average intensity (A.U.)")

        peaks, properties = find_peaks(counts, height=height[frame], width=3)
        x = bins[peaks[0]]
        y = counts[peaks[0]]
        fit_results = fit_gaussian(bins, counts, x)
        fwhm_over_radius.append(fit_results[0])
        popt = fit_results[1]
        r_squared = fit_results[2]
        x_fit = np.arange(x - 20, x + 20, 1)
        y_fit = gaussian_lin(x_fit, *popt)

        plt.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n r:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM/r : {round(fwhm_over_radius[-1],3)}",
        )
        ax.scatter(x, y, c="r", marker="X", s=100)
        ax.set_title(f"Sol67 {round(frame*0.3,1)} A")
        plt.vlines([x - 20, x + 20], 0, 500, "r")
        ax.legend()
        plt.savefig(f"{output}/radial_average/mos_magnet_scan_{frame}.png")
        plt.show()

    current = np.arange(0, 3.1, 0.3)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(current, fwhm_over_radius, c="b")
    ax.set_title(f"Sol67 optimization")
    ax.set_xlabel("Current Sol67(A)")
    ax.set_ylabel("FWHM/r")
    plt.savefig(f"{output}/radial_average/mos_magnet_scan_optmize.png")
    plt.show()


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot resolution rings. Parameters need to be correctly set in code."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="input image padded"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output folder images and gif"
    )
    parser.add_argument(
        "-l", "--label", type=str, action="store", help="output folder images and gif"
    )
    args = parser.parse_args()
    file_path = args.input
    output_folder = args.output
    get_minimum(file_path, output_folder)


if __name__ == "__main__":
    main()
