#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
import random
from datetime import datetime
import os

sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
import geometry_funcs as gf
import argparse
import numpy as np
from utils import (
    get_format,
    open_fwhm_map,
    open_fwhm_map_global_min,
    gaussian_lin,
    azimuthal_average,
    center_of_mass
)
import skimage.filters

from PIL import Image
import itertools
import multiprocessing
import math
import matplotlib.pyplot as plt
import subprocess as sub
from scipy.optimize import curve_fit
import h5py
import om.utils.crystfel_geometry as crystfel_geometry
from scipy.signal import find_peaks as find_peaks

Frame = 15
ForceCenter = [592, 533]
PeakIndex = 0
AutoFlag=True
Width = 10

def label_intensity(image, sigma=0.1, min_value=100, max_value=200):

    # denoise the image with a Gaussian filters
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)
    # mask the image according to threshold
    binary_mask=np.ones(image.shape, dtype=bool)
    binary_mask[np.where(blurred_image<min_value)] = False
    binary_mask[np.where(blurred_image>max_value)] = False

    print(type(binary_mask[0,0]), binary_mask[0,0])
    # perform connected component analysis
    labeled_image = binary_mask.astype(int)
    return labeled_image


def apply_geom(data: np.ndarray, geometry_filename: str) -> np.ndarray:
    ## Apply crystfel geomtry file .geom
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2 * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min()))) + 2
    )
    x_minimum: int = (
        2 * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min()))) + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    corr_data = crystfel_geometry.apply_geometry_to_data(data, geometry)
    return corr_data


def calculate_fwhm(data_and_coordinates: tuple) -> Dict[str, int]:
    corrected_data, mask, center_to_radial_average = data_and_coordinates
    x, y = azimuthal_average(corrected_data, center=center_to_radial_average, mask=mask)
    x_all = x.copy()
    y_all = y.copy()
    # Plot all radial average
    if plot_flag:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(x, y)

    ## Define powder ring peak region
    peaks, properties = find_peaks(y, height=80, width=5)
    if len(peaks)>PeakIndex:
        peak_position = peaks[PeakIndex]
        print(peak_position)
        a = y[peak_position]
        x = x[peak_position - Width : peak_position + Width]
        y = y[peak_position - Width : peak_position + Width]

        m0 = (y[-1] - y[0]) / (x[-1] - x[0])
        n0 = ((y[-1] + y[0]) - m0 * (x[-1] + x[0])) / 2
        y_linear = m0 * x + n0
        y_gaussian = y - y_linear

        mean = sum(x * y_gaussian) / sum(y_gaussian)
        sigma = np.sqrt(sum(y_gaussian * (x - mean) ** 2) / sum(y_gaussian))
        try:
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
        except:
            r_squared = 1
            fwhm = 800
            fwhm_over_radius = 800
            popt = []
    else:
        r_squared = 1
        fwhm = 8000
        fwhm_over_radius = 8000
        popt = []
    ## Display plots
    if plot_flag and len(popt) > 0:
        x_fit = x.copy()
        y_fit = gaussian_lin(x_fit, *popt)

        plt.vlines([x[0], x[-1]], 0, 3 * round(popt[0]), "r")

        plt.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
        )
        plt.title("Azimuthal integration")
        plt.xlim(0, 350)
        plt.ylim(0, 2 * round(popt[0]))
        plt.legend()
        plt.savefig(
            f"{args.output}/plots/gaussian_fit/{stamp}_{center_to_radial_average[0]}_{center_to_radial_average[1]}.png"
        )
        plt.close()

    return {
        "xc": center_to_radial_average[0],
        "yc": center_to_radial_average[1],
        "fwhm": fwhm,
        "fwhm_over_radius": fwhm_over_radius,
        "r_squared": r_squared,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of diffraction patterns fro MHz beam sweeping serial crystallography."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-m", "--mask", type=str, action="store", help="path to list of mask files .lst"
    )

    parser.add_argument(
        "-g",
        "--geom",
        type=str,
        action="store",
        help="CrystFEL geometry filename",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )
    global args
    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    if args.mask:
        mask_files = open(args.mask, "r")
        mask_paths = mask_files.readlines()
        mask_files.close()
    else:
        mask_paths = []

    file_format = get_format(args.input)

    ### Extract geometry file
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        args.geom, return_dict=True
    )
    y_minimum: int = 2 * int(max(abs(y_map.max()), abs(y_map.min()))) + 2
    x_minimum: int = 2 * int(max(abs(x_map.max()), abs(x_map.min()))) + 2
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    preamb, dim_info = gf.read_geometry_file_preamble(args.geom)
    dist_m = preamb["coffset"]
    res = preamb["res"]
    clen = preamb["clen"]
    dist = 0.0

    # print('Det',det_dict)

    # global DetectorCenter

    # DetectorCenter=[_img_center_x, _img_center_y]

    #print(DetectorCenter)
    # output_folder = os.path.dirname(args.output)

    label = (
        (args.output).split("/")[-1]
        + "_"
        + ((args.input).split("/")[-1]).split(".")[-1][3:]
    )
    global stamp
    global plot_flag
    plot_flag = False

    if file_format == "lst":
        for i in range(0, len(paths[:])):

        #for i in range(Frame, Frame + 1):
            stamp = f"magnet_scan_{i}"

            file_name = paths[i][:-1]
            if len(mask_paths) > 0:
                mask_file_name = mask_paths[0][:-1]
            else:
                mask_file_name = False

            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
            elif get_format(file_name) == "h":
                f = h5py.File(f"{file_name}", "r")
                data = np.array(f["data"])
                f.close()
            elif get_format(file_name) == "tif":
                data = np.array(Image.open(file_name))

            if not mask_file_name:
                mask = np.ones(data.shape)
            else:
                if get_format(mask_file_name) == "cbf":
                    xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                    # Mask of defective pixels
                    xds_mask[np.where(xds_mask <= 0)] = 0
                    xds_mask[np.where(xds_mask > 0)] = 1
                    # Mask hot pixels
                    xds_mask[np.where(data > 1e5)] = 0
                    mask = xds_mask
                elif get_format(mask_file_name) == "h":
                    f = h5py.File(f"{mask_file_name}", "r")
                    mask = np.array(f["data/data"])
                    mask = np.array(mask, dtype=np.int32)
                    mask = apply_geom(mask, args.geom)
                    f.close()

            corrected_data = data
            labeled = label_intensity(corrected_data*mask, sigma=0.1, min_value=160, max_value=1000)
            roi=labeled.copy()

            # Mask of defective pixels
            mask[np.where(corrected_data < 0)] = 0
            
            now = datetime.now()
            print(f"Current begin time = {now}")
            

            if AutoFlag:
                first_center = center_of_mass(corrected_data, mask)
            else:
                first_center = ForceCenter.copy()
            
            ## Grid search of shifts around the detector center

            pixel_step = 1
            xx, yy = np.meshgrid(
                np.arange(
                    first_center[0] - 5, first_center[0] + 6, pixel_step, dtype=int
                ),
                np.arange(
                    first_center[1] - 5, first_center[1] + 6, pixel_step, dtype=int
                ),
            )

            coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
            coordinates_anchor_data = [
                (corrected_data, mask*roi, shift) for shift in coordinates
            ]

            fwhm_summary = []
            for shift in coordinates_anchor_data:
                fwhm_summary.append(calculate_fwhm(shift))
           
            xc, yc = open_fwhm_map_global_min(
                fwhm_summary, args.output, f"{label}_{i}", pixel_step
            )
            
            refined_center = (np.around(xc, 1), np.around(yc, 1))

            plot_flag = True
            results = calculate_fwhm((corrected_data, mask, (xc, yc)))
            plot_flag = False

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            pos = ax.imshow(corrected_data * mask, vmax=200, cmap="cividis")
            ax.scatter(
                first_center[0],
                first_center[1],
                color="lime",
                marker="+",
                s=150,
                label=f"Initial center:({first_center[0]},{first_center[1]})",
            )
            ax.scatter(
                refined_center[0],
                refined_center[1],
                color="r",
                marker="o",
                s=25,
                label=f"Refined center:({refined_center[0]}, {refined_center[1]})",
            )
            ax.set_xlim(100, 1000)
            ax.set_ylim(1000, 100)
            plt.title("Center refinement: FWHM minimization")
            fig.colorbar(pos, shrink=0.6)
            ax.legend()
            plt.savefig(f"{args.output}/plots/centered/{label}_{i}.png")
            plt.close()

            if args.output:
                f = h5py.File(f"{args.output}/h5_files/{label}_{i}.h5", "w")
                f.create_dataset("hit", data=1)
                f.create_dataset("id", data=file_name)
                f.create_dataset("intensity", data=np.sum(corrected_data * mask))
                f.create_dataset("refined_center", data=refined_center)
            now = datetime.now()
            print(f"Current end time = {now}")


if __name__ == "__main__":
    main()
