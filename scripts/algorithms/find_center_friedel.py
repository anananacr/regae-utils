#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
from datetime import datetime
import os

sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
import geometry_funcs as gf
import argparse
import numpy as np
from utils import (
    get_format,
    open_distance_map_global_min,
    center_of_mass,
    circle_mask,
    open_distance_map_fit_min,
)
from PIL import Image
from models import PF8, PF8Info
import math
import matplotlib.pyplot as plt
import h5py
import om.utils.crystfel_geometry as crystfel_geometry

DetectorCenter = [606, 539]
MinPeaks = 4
SearchRadius = 8
AutoFlag = False
OuterMask = True
OuterRadius = 200
OutlierDistance = 4

PF8Config = PF8Info(
    max_num_peaks=10000,
    adc_threshold=0,
    minimum_snr=5,
    min_pixel_count=4,
    max_pixel_count=1000,
    local_bg_radius=10,
    min_res=10,
    max_res=180,
)

BeamSweepingParam = {
    "detector_center": DetectorCenter,
    "min_peaks": MinPeaks,
    "search_radius": SearchRadius,
    "pf8_max_num_peaks": PF8Config.max_num_peaks,
    "pf8_adc_threshold": PF8Config.adc_threshold,
    "pf8_minimum_snr": PF8Config.minimum_snr,
    "pf8_min_pixel_count": PF8Config.min_pixel_count,
    "pf8_max_pixel_count": PF8Config.max_pixel_count,
    "pf8_local_bg_radius": PF8Config.local_bg_radius,
    "pf8_min_res": PF8Config.min_res,
    "pf8_max_res": PF8Config.max_res,
    "auto_flag": AutoFlag,
    "outer_mask": OuterMask,
    "outlier_distance": OutlierDistance,
}


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

    assembled_data = crystfel_geometry.apply_geometry_to_data(data, geometry)
    return assembled_data


def remove_repeated_items(pairs_list: list) -> list:
    x_vector = []
    y_vector = []
    unique_pairs = []

    for pair in pairs_list:
        peak_0, peak_1 = pair
        x = peak_0[0] - peak_1[0]
        y = peak_0[1] - peak_1[1]
        if x not in x_vector and y not in y_vector:
            x_vector.append(x)
            y_vector.append(y)
            unique_pairs.append((peak_0, peak_1))
    return unique_pairs


def shift_inverted_peaks_and_calculate_minimum_distance(
    peaks_and_shift: list,
) -> Dict[str, float]:
    peaks_list, inverted_peaks, shift = peaks_and_shift
    shifted_inverted_peaks = [(x + shift[0], y + shift[1]) for x, y in inverted_peaks]
    distance = calculate_pair_distance(peaks_list, shifted_inverted_peaks)

    return {
        "shift_x": shift[0],
        "xc": (shift[0] / 2) + initial_center[0],
        "shift_y": shift[1],
        "yc": (shift[1] / 2) + initial_center[1],
        "d": distance,
    }


def calculate_pair_distance(peaks_list: list, shifted_peaks_list: list) -> float:
    d = [
        math.sqrt((peaks_list[idx][0] - i[0]) ** 2 + (peaks_list[idx][1] - i[1]) ** 2)
        for idx, i in enumerate(shifted_peaks_list)
    ]
    return sum(d)


def select_closest_peaks(peaks_list: list, inverted_peaks: list) -> list:
    peaks = []
    for i in inverted_peaks:
        radius = 1
        found_peak = False
        while not found_peak and radius <= SearchRadius:
            found_peak = find_a_peak_in_the_surrounding(peaks_list, i, radius)
            radius += 1
        if found_peak:
            peaks.append((found_peak, i))
    peaks = remove_repeated_items(peaks)
    return peaks


def find_a_peak_in_the_surrounding(
    peaks_list: list, inverted_peak: list, radius: int
) -> list:
    cut_peaks_list = []
    cut_peaks_list = [
        (
            peak,
            math.sqrt(
                (peak[0] - inverted_peak[0]) ** 2 + (peak[1] - inverted_peak[1]) ** 2
            ),
        )
        for peak in peaks_list
        if math.sqrt(
            (peak[0] - inverted_peak[0]) ** 2 + (peak[1] - inverted_peak[1]) ** 2
        )
        <= radius
    ]
    cut_peaks_list.sort(key=lambda x: x[1])

    if cut_peaks_list == []:
        return False
    else:
        return cut_peaks_list[0][0]


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

    print(DetectorCenter)
    output_folder = os.path.dirname(args.output)
    global initial_center
    initial_center = [0, 0]

    label = (
        (args.output).split("/")[-1]
        + "_"
        + ((args.input).split("/")[-1]).split(".")[-1][3:]
    )

    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):

            file_name = paths[i][:-1]
            if len(mask_paths) > 0:
                mask_file_name = mask_paths[0][:-1]
            else:
                mask_file_name = False

            frame_number = i
            print(file_name)
            label = (file_name.split("/")[-1]).split(".")[0]

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
            ## Polarization correction factor
            """
            corrected_data, pol_array_first = correct_polarization(
                x_map, y_map, clen_v, data, mask=mask
            )
            """
            corrected_data = data
            h, w = corrected_data.shape
            print(len(corrected_data))

            # Mask of defective pixels
            mask[np.where(corrected_data < 0)] = 0
            # Mask hot pixels
            # mask[np.where(corrected_data > 1e5)] = 0

            if AutoFlag:
                if OuterMask:
                    outer_mask = circle_mask(
                        corrected_data, DetectorCenter, OuterRadius
                    )
                    initial_center = center_of_mass(corrected_data, mask * outer_mask)
                else:
                    initial_center = center_of_mass(corrected_data, mask)

                distance = math.sqrt(
                    (initial_center[0] - DetectorCenter[0]) ** 2
                    + (initial_center[1] - DetectorCenter[1]) ** 2
                )
                if distance > OutlierDistance:
                    initial_center = DetectorCenter
            else:
                initial_center = DetectorCenter
            ## Peakfinder8 detector information and bad_pixel_map

            PF8Config.pf8_detector_info = dict(
                asic_nx=mask.shape[1],
                asic_ny=mask.shape[0],
                nasics_x=1,
                nasics_y=1,
            )
            PF8Config._bad_pixel_map = mask
            PF8Config.modify_radius(initial_center[0], initial_center[1])
            pf8 = PF8(PF8Config)
            peaks_list = pf8.get_peaks_pf8(data=corrected_data)
            if peaks_list["num_peaks"] > MinPeaks:
                now = datetime.now()
                print(f"Current begin time = {now}")
                peaks_list_x = [k - initial_center[0] for k in peaks_list["fs"]]
                peaks_list_y = [k - initial_center[1] for k in peaks_list["ss"]]
                peaks = list(zip(peaks_list_x, peaks_list_y))

                inverted_peaks_x = [-1 * k for k in peaks_list_x]
                inverted_peaks_y = [-1 * k for k in peaks_list_y]
                inverted_peaks = list(zip(inverted_peaks_x, inverted_peaks_y))
                pairs_list = select_closest_peaks(peaks, inverted_peaks)

                ## Grid search of shifts around the detector center
                pixel_step = 0.2
                xx, yy = np.meshgrid(
                    np.arange(-30, 30.2, pixel_step, dtype=float),
                    np.arange(-30, 30.2, pixel_step, dtype=float),
                )
                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                peaks_0 = [x for x, y in pairs_list]
                peaks_1 = [y for x, y in pairs_list]
                coordinates_anchor_peaks = [
                    [peaks_0, peaks_1, shift] for shift in coordinates
                ]
                distance_summary = []
                for shift in coordinates_anchor_peaks:
                    distance_summary.append(
                        shift_inverted_peaks_and_calculate_minimum_distance(shift)
                    )
                ## Display plots

                ## Fine tune
                # xc, yc, converged = open_distance_map_global_min(
                #    distance_summary, output_folder, f"{label}", pixel_step
                # )
                ## Ultrafine tune
                xc, yc, converged = open_distance_map_fit_min(
                    distance_summary, output_folder, f"{label}", pixel_step
                )

                refined_center = (np.around(xc, 1), np.around(yc, 1))
                shift_x = 2 * (xc - initial_center[0])
                shift_y = 2 * (yc - initial_center[1])

                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                pos = ax.imshow(corrected_data * mask, vmax=200, cmap="cividis")
                ax.scatter(
                    initial_center[0],
                    initial_center[1],
                    color="lime",
                    marker="+",
                    s=150,
                    label=f"Initial center:({np.round(initial_center[0],1)},{np.round(initial_center[1], 1)})",
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
                plt.title("Center refinement: autocorrelation of Friedel pairs")
                fig.colorbar(pos, shrink=0.6)
                ax.legend()
                plt.savefig(f"{output_folder}/plots/centered/{label}.png")
                plt.close("all")

                original_peaks_x = [
                    np.round(k + initial_center[0]) for k in peaks_list_x
                ]
                original_peaks_y = [
                    np.round(k + initial_center[1]) for k in peaks_list_y
                ]
                inverted_non_shifted_peaks_x = [
                    np.round(k + initial_center[0]) for k in inverted_peaks_x
                ]
                inverted_non_shifted_peaks_y = [
                    np.round(k + initial_center[1]) for k in inverted_peaks_y
                ]
                inverted_shifted_peaks_x = [
                    np.round(k + initial_center[0] + shift_x) for k in inverted_peaks_x
                ]
                inverted_shifted_peaks_y = [
                    np.round(k + initial_center[1] + shift_y) for k in inverted_peaks_y
                ]

                ## Check pairs alignement
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                pos = ax.imshow(corrected_data * mask, vmax=200, cmap="cividis")
                ax.scatter(
                    original_peaks_x,
                    original_peaks_y,
                    facecolor="none",
                    s=80,
                    marker="s",
                    edgecolor="red",
                    label="original peaks",
                )
                ax.scatter(
                    inverted_non_shifted_peaks_x,
                    inverted_non_shifted_peaks_y,
                    s=80,
                    facecolor="none",
                    marker="D",
                    edgecolor="blue",
                    label="inverted peaks",
                )
                ax.scatter(
                    inverted_shifted_peaks_x,
                    inverted_shifted_peaks_y,
                    facecolor="none",
                    s=50,
                    marker="D",
                    edgecolor="lime",
                    label="shift of inverted peaks",
                )

                ax.set_xlim(100, 1000)
                ax.set_ylim(1000, 100)
                plt.title("Bragg peaks alignement")
                fig.colorbar(pos, shrink=0.6)
                ax.legend()
                plt.savefig(f"{output_folder}/plots/peaks/{label}.png")
                plt.close()

                original_peaks_x = [k + initial_center[0] for k in peaks_list_x]
                original_peaks_y = [k + initial_center[1] for k in peaks_list_y]
                inverted_non_shifted_peaks_x = [
                    k + initial_center[0] for k in inverted_peaks_x
                ]
                inverted_non_shifted_peaks_y = [
                    k + initial_center[1] for k in inverted_peaks_y
                ]
                inverted_shifted_peaks_x = [
                    k + initial_center[0] + shift_x for k in inverted_peaks_x
                ]
                inverted_shifted_peaks_y = [
                    k + initial_center[1] + shift_y for k in inverted_peaks_y
                ]
                if args.output:
                    f = h5py.File(f"{output_folder}/h5_files/{label}.h5", "w")
                    grp = f.create_group("data")
                    grp.create_dataset("hit", data=1)
                    grp.create_dataset("converged", data=converged)
                    grp.create_dataset("id", data=file_name)
                    grp.create_dataset("intensity", data=np.sum(corrected_data * mask))
                    if converged == 1:
                        grp.create_dataset("refined_center", data=refined_center)
                    else:
                        grp.create_dataset(
                            "refined_center",
                            data=[initial_center[0] - 20, initial_center[1] - 20],
                        )
                    grp = f.create_group("beam_sweeping_config")
                    for key, value in BeamSweepingParam.items():
                        grp.create_dataset(key, data=value)
                    grp.create_dataset("initial_center", data=initial_center)
                    grp = f.create_group("peaks_positions")
                    grp.create_dataset("original_peaks_x", data=original_peaks_x)
                    grp.create_dataset("original_peaks_y", data=original_peaks_y)
                    grp.create_dataset(
                        "inverted_peaks_x", data=inverted_non_shifted_peaks_x
                    )
                    grp.create_dataset(
                        "inverted_peaks_y", data=inverted_non_shifted_peaks_y
                    )
                    grp.create_dataset("shifted_peaks_x", data=inverted_shifted_peaks_x)
                    grp.create_dataset("shifted_peaks_y", data=inverted_shifted_peaks_y)
            else:
                f = h5py.File(f"{output_folder}/h5_files/{label}.h5", "w")
                grp = f.create_group("data")
                grp.create_dataset("hit", data=0)
                grp.create_dataset("converged", data=0)
                grp.create_dataset("id", data=file_name)
                grp.create_dataset("intensity", data=np.sum(corrected_data * mask))
                grp.create_dataset(
                    "refined_center",
                    data=initial_center,
                )
                grp = f.create_group("beam_sweeping_config")
                for key, value in BeamSweepingParam.items():
                    grp.create_dataset(key, data=value)

                now = datetime.now()
                print(f"Current end time = {now}")


if __name__ == "__main__":
    main()
