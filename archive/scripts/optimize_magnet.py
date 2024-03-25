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
from numpy import exp
from utils import azimuthal_average, gaussian
from scipy.signal import find_peaks as find_peaks
from scipy.optimize import curve_fit

max_frames = 20
increment_current = 0.1
height_base = 80

initial_guess_x = [
    553,
    553,
    553,
    553,
    553,
    553,
    553,
    553,
    553,
    553,
    553,
    553,
    555,
    559,
    559,
    559,
    559,
    559,
    559,
    559,
    559,
    555,
    555,
    555,
    555,
    555,
    555,
    555,
    555,
    555,
    555,
]

initial_guess_y = [
    534,
    534,
    534,
    534,
    533,
    533,
    533,
    533,
    533,
    533,
    533,
    533,
    533,
    535,
    535,
    535,
    535,
    535,
    535,
    535,
    535,
    535,
    535,
    533,
    533,
    533,
    533,
    533,
    533,
    533,
    533,
]

def get_minimum(file_path: str, output: str):
    ## Rings radius in pixels
    rings = [10]
    ## Center of the image [xc,yc]

    frames = np.arange(0, max_frames + 1, 1)
    fwhm_over_radius = []

    global frame_number

    for frame in frames:
        frame_number = frame
        data = np.array(Image.open(f"{file_path}_{frame:06}.tif"))
        mask = np.ones_like(data)
        mask[np.where(data == -1)] = 0
        bins, counts = azimuthal_average(data, center[frame], mask)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(bins, counts)
        ax.set_ylim(0, 600)
        ax.set_xlim(0, 350)
        ax.set_xlabel("Radial distance (pixel)")
        ax.set_ylabel("Average intensity (A.U.)")

        peaks, properties = find_peaks(counts, height=height[frame], width=5)
        print(f"{round(frame*increment_current,1)} A", peaks)
        x = bins[peaks[1]]
        y = counts[peaks[1]]
        fit_results = fit_gaussian(bins, counts, x)
        fwhm_over_radius.append(fit_results[0])
        popt = fit_results[1]
        r_squared = fit_results[2]
        if frame_number < 12:
            right_leg = 16  # for peak_3
            #right_leg=24
        else:
            right_leg = 8  # for peak_3
            #right_leg=20
        x_fit = np.arange(x - 10, x + right_leg, 1)
        y_fit = gaussian_lin(x_fit, *popt)

        plt.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n r:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM/r : {round(fwhm_over_radius[-1],3)}",
        )
        ax.scatter(x, y, c="r", marker="X", s=100)
        ax.set_title(f"Sol67 {round(2+(frame*increment_current),1)} A")
        plt.vlines([x - 10, x + right_leg], 0, 500, "r")
        ax.legend()
        plt.savefig(f"{output}/plots/radial_average/au_magnet_scan_{frame}.png")
        plt.show()

    current = np.arange(0, increment_current * (max_frames + 1), increment_current)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(current, fwhm_over_radius, c="b")
    ax.set_title(f"Sol67 optimization")
    ax.set_xlabel("Current Sol67(A)")
    ax.set_ylabel("FWHM/r")
    plt.savefig(f"{output}/plots/radial_average/au_magnet_scan_optmize.png")
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

    global center
    center = np.zeros((max_frames, 2), dtype=np.int32)
    global height
    height = height_base * np.ones(max_frames + 1)
    height_adjusted = [x + 50 if idx >= 18 else x for idx, x in enumerate(height)]
    height = height_adjusted.copy()
    print(height)
    current = np.arange(2, 2+ (max_frames + 1) * increment_current, increment_current)
    print(current)
    x = current
    y = initial_guess_x
    plt.scatter(initial_guess_x, initial_guess_y)
    plt.show()
    plt.scatter(current, initial_guess_x[: max_frames + 1])
    plt.scatter(current, initial_guess_y[: max_frames + 1])
    plt.show()
    center = list(
        zip(initial_guess_x[: max_frames + 1], initial_guess_y[: max_frames + 1])
    )
    args = parser.parse_args()
    file_path = args.input
    output_folder = args.output
    get_minimum(file_path, output_folder)


if __name__ == "__main__":
    main()
