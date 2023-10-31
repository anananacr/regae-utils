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
from utils import get_format, update_corner_in_geom, open_distance_map_global_min
from PIL import Image
import itertools
from models import PF8, PF8Info
import multiprocessing
import math
import matplotlib.pyplot as plt
import subprocess as sub
import h5py
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import om.utils.crystfel_geometry as crystfel_geometry


DetectorCenter = [586, 533]
MinPeaks = 2
global pf8_info

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=50,
    minimum_snr=3,
    min_pixel_count=4,
    max_pixel_count=1000,
    local_bg_radius=10,
    min_res=80,
    max_res=350,
)


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


def get_peaks_on_quadrant(peaks: list, n: int) -> list:
    peaks_on_quadrants = []
    for i in peaks:
        if i[0] >= 0:
            if i[1] >= 0 and n == 1:
                peaks_on_quadrants.append(i)
            elif i[1] < 0 and n == 4:
                peaks_on_quadrants.append(i)
        else:
            if i[1] >= 0 and n == 2:
                peaks_on_quadrants.append(i)
            elif i[1] < 0 and n == 3:
                peaks_on_quadrants.append(i)
    return peaks_on_quadrants


def take_pairs_with_minimum_distance(a, b):
    permutations = list(itertools.permutations(b))
    distances = []
    pairs = []
    for permutation in permutations:
        d = [
            math.sqrt((a[idx][0] - i[0]) ** 2 + (a[idx][1] - i[1]) ** 2)
            for idx, i in enumerate(list(permutation))
        ]
        p = [
            [(a[idx][0], a[idx][1]), (i[0], i[1])]
            for idx, i in enumerate(list(permutation))
        ]
        distances.append(d)
        pairs.append(p)
    distances = np.array(distances)
    minimum_distance = np.min(distances)
    pairs = np.array(pairs)
    index = np.where(distances <= minimum_distance + 10)
    coordinates = list(zip(index[0], index[1]))
    pairs_list = [list(pairs[i]) for i in coordinates]
    return pairs_list


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


def sort_by_minimum_distance(a: list, b: list) -> list:
    permutations = list(itertools.permutations(b))
    distances = []
    for permutation in permutations:
        d = [
            math.sqrt((a[idx][0] - i[0]) ** 2 + (a[idx][1] - i[1]) ** 2)
            for idx, i in enumerate(list(permutation))
        ]
        distances.append(sum(d))
    index = np.argmin(distances)
    return permutations[index]


def shift_inverted_peaks_and_calculate_minimum_distance(
    peaks_and_shift: list,
) -> Dict[str, float]:
    peaks_list, inverted_peaks, shift = peaks_and_shift
    shifted_inverted_peaks = [(x + shift[0], y + shift[1]) for x, y in inverted_peaks]
    distance = calculate_pair_distance(peaks_list, shifted_inverted_peaks)

    return {
        "shift_x": shift[0],
        "xc": (shift[0] / 2) + DetectorCenter[0],
        "shift_y": shift[1],
        "yc": (shift[1] / 2) + DetectorCenter[1],
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
        while not found_peak and radius <= 20:
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

    print(DetectorCenter)
    output_folder = os.path.dirname(args.output)

    label = (
        (args.output).split("/")[-1]
        + "_"
        + ((args.input).split("/")[-1]).split(".")[-1][3:]
    )
    """
    if clen is not None:
        if not gf.is_float_try(clen):
            check = H5_name + clen
            myCmd = os.popen("h5ls " + check).read()
            if "NOT" in myCmd:
                # print("Error: no clen from .h5 file")
                clen_v = 0.0
            else:
                f = h5py.File(H5_name, "r")
                clen_v = f[clen][()] * (1e-3)  # f[clen].value * (1e-3)
                f.close()
                pol_bool = True
                # print("Take into account polarisation")
        else:
            clen_v = float(clen)
            pol_bool = True
            # print("Take into account polarisation")

        if dist_m is not None:
            dist_m += clen_v
        else:
            # print("Error: no coffset in geometry file. It is considered as 0.")
            dist_m = 0.0
        # print("CLEN, COFSET", clen, dist_m)
        dist = dist_m * res
    """

    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):
            # for i in range(0, 20):

            file_name = paths[i][:-1]
            if len(mask_paths) > 0:
                mask_file_name = mask_paths[0][:-1]
            else:
                mask_file_name = False

            frame_number = i
            print(file_name)

            initial_center = DetectorCenter

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

            # for n_frame,frame in enumerate(data):
            for k in range(1):

                ## Polarization correction factor
                # corrected_data=frame
                corrected_data = data
                h, w = corrected_data.shape
                print(len(corrected_data))
                # Mask of defective pixels
                mask[np.where(corrected_data < 0)] = 0
                # Mask hot pixels
                # mask[np.where(corrected_data > 1e5)] = 0
                """
                corrected_data, pol_array_first = correct_polarization(
                    x_map, y_map, clen_v, data, mask=mask
                )
                """
                ## Peakfinder8 detector information and bad_pixel_map

                pf8_info.pf8_detector_info = dict(
                    asic_nx=mask.shape[1],
                    asic_ny=mask.shape[0],
                    nasics_x=1,
                    nasics_y=1,
                )
                pf8_info._bad_pixel_map = mask
                pf8_info.modify_radius(DetectorCenter[0], DetectorCenter[1])
                pf8 = PF8(pf8_info)
                peaks_list = pf8.get_peaks_pf8(data=corrected_data)

                if peaks_list["num_peaks"] > MinPeaks:
                    now = datetime.now()
                    print(f"Current begin time = {now}")

                    peaks_list_x = [k - DetectorCenter[0] for k in peaks_list["fs"]]
                    peaks_list_y = [k - DetectorCenter[1] for k in peaks_list["ss"]]

                    peaks = list(zip(peaks_list_x, peaks_list_y))
                    peaks_list_x = []
                    peaks_list_y = []
                    for k in range(1, 5):
                        peaks_on_quadrant = get_peaks_on_quadrant(peaks, k)
                        peaks_on_quadrant.sort(
                            key=lambda x: math.sqrt(x[0] ** 2 + x[1] ** 2)
                        )
                        peaks_on_quadrant_x = [x for x, y in peaks_on_quadrant[:]]
                        peaks_on_quadrant_y = [y for x, y in peaks_on_quadrant[:]]
                        peaks_list_x += peaks_on_quadrant_x
                        peaks_list_y += peaks_on_quadrant_y

                    peaks = list(zip(peaks_list_x, peaks_list_y))

                    inverted_peaks_x = [-1 * k for k in peaks_list_x]
                    inverted_peaks_y = [-1 * k for k in peaks_list_y]
                    inverted_peaks = list(zip(inverted_peaks_x, inverted_peaks_y))

                    pairs_list = select_closest_peaks(peaks, inverted_peaks)
                    print("List of pairs", pairs_list)

                    ## Grid search of shifts around the detector center
                    pixel_step = 0.2
                    xx, yy = np.meshgrid(
                        np.arange(-20, 20.2, pixel_step, dtype=float),
                        np.arange(-20, 20.2, pixel_step, dtype=float),
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
                    xc, yc = open_distance_map_global_min(
                        distance_summary,
                        output_folder,
                        f"{label}_{i}_{frame_number}",
                        pixel_step,
                    )

                    refined_center = (np.around(xc, 1), np.around(yc, 1))
                    shift_x = 2 * (xc - DetectorCenter[0])
                    shift_y = 2 * (yc - DetectorCenter[1])
                    print("center", refined_center)

                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    pos = ax.imshow(corrected_data * mask, vmax=200, cmap="cividis")
                    ax.scatter(
                        DetectorCenter[0],
                        DetectorCenter[1],
                        color="lime",
                        marker="+",
                        s=150,
                        label=f"Initial center:({DetectorCenter[0]},{DetectorCenter[1]})",
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
                    plt.savefig(f"{output_folder}/plots/centered/{label}_{i}.png")
                    # plt.savefig(f"{args.output}/plots/centered/{label}_{i}_{n_frame}.png")
                    # plt.show()
                    plt.close()

                    original_peaks_x = [
                        np.round(k + DetectorCenter[0]) for k in peaks_list_x
                    ]
                    original_peaks_y = [
                        np.round(k + DetectorCenter[1]) for k in peaks_list_y
                    ]
                    inverted_non_shifted_peaks_x = [
                        np.round(k + DetectorCenter[0]) for k in inverted_peaks_x
                    ]
                    inverted_non_shifted_peaks_y = [
                        np.round(k + DetectorCenter[1]) for k in inverted_peaks_y
                    ]
                    inverted_shifted_peaks_x = [
                        np.round(k + DetectorCenter[0] + shift_x)
                        for k in inverted_peaks_x
                    ]
                    inverted_shifted_peaks_y = [
                        np.round(k + DetectorCenter[1] + shift_y)
                        for k in inverted_peaks_y
                    ]

                    ## Check pairs allignement
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
                    # if peaks_0!=[]:
                    #    peak_x_slab=[k+DetectorCenter[0] for k in peaks_0
                    #    ax.scatter((peaks_0[0]+DetectorCenter[0],-1*peak_1[0]+DetectorCenter[0]), (peak_0[1]+DetectorCenter[1],-1*peak_1[1]+DetectorCenter[1]), facecolor="none",  s=200,marker= '*', edgecolor="magenta", label='friedel pair')

                    ax.set_xlim(100, 1000)
                    ax.set_ylim(1000, 100)
                    plt.title("Bragg peaks allignement")
                    fig.colorbar(pos, shrink=0.6)
                    ax.legend()
                    plt.savefig(f"{output_folder}/plots/peaks/{label}_{i}.png")
                    # plt.savefig(f"{args.output}/plots/peaks/{label}_{i}_{n_frame}.png")
                    # plt.show()
                    plt.close()
                    """
                    # Update geom 
                    updated_geom = f"{args.geom[:-5]}_{label}_{i}_v1.geom"
                    cmd = f"cp {args.geom} {updated_geom}"
                    sub.call(cmd, shell=True)
                    update_corner_in_geom(updated_geom, xc, yc)
                    cmd = f"cp {updated_geom} {output_folder}/final_geom "
                    sub.call(cmd, shell=True)
                    
                    ## Clean geom directory
                    updated_geom = f"{args.geom[:-5]}_{label}_{i}_v1.geom"
                    cmd = f"cp {updated_geom} {output_folder}"
                    sub.call(cmd, shell=True)
                    root_directory = os.path.dirname(args.geom)
                    cmd = f"rm {root_directory}/*{label}*.geom"
                    sub.call(cmd, shell=True)
                    cmd = (
                        f"mv {output_folder}/*v1.geom {root_directory}"
                    )
                    sub.call(cmd, shell=True)
                    """

                    original_peaks_x = [k + DetectorCenter[0] for k in peaks_list_x]
                    original_peaks_y = [k + DetectorCenter[1] for k in peaks_list_y]
                    inverted_non_shifted_peaks_x = [
                        k + DetectorCenter[0] for k in inverted_peaks_x
                    ]
                    inverted_non_shifted_peaks_y = [
                        k + DetectorCenter[1] for k in inverted_peaks_y
                    ]
                    inverted_shifted_peaks_x = [
                        k + DetectorCenter[0] + shift_x for k in inverted_peaks_x
                    ]
                    inverted_shifted_peaks_y = [
                        k + DetectorCenter[1] + shift_y for k in inverted_peaks_y
                    ]

                    if args.output:
                        f = h5py.File(f"{output_folder}/h5_files/{label}_{i}.h5", "w")
                        # f = h5py.File(f"{output_folder}/h5_files/{label}_{i}_{n_frame}.h5", "w")
                        f.create_dataset("hit", data=1)
                        f.create_dataset("id", data=file_name)
                        # f.create_dataset("index", data=n_frame)
                        f.create_dataset(
                            "intensity", data=np.sum(corrected_data * mask)
                        )
                        f.create_dataset("refined_center", data=refined_center)
                        f.create_dataset("original_peaks_x", data=original_peaks_x)
                        f.create_dataset("original_peaks_y", data=original_peaks_y)
                        f.create_dataset(
                            "inverted_peaks_x", data=inverted_non_shifted_peaks_x
                        )
                        f.create_dataset(
                            "inverted_peaks_y", data=inverted_non_shifted_peaks_y
                        )
                        f.create_dataset(
                            "shifted_peaks_x", data=inverted_shifted_peaks_x
                        )
                        f.create_dataset(
                            "shifted_peaks_y", data=inverted_shifted_peaks_y
                        )
                else:
                    f = h5py.File(f"{output_folder}/h5_files/{label}_{i}.h5", "w")
                    # f = h5py.File(f"{output_folder}/h5_files/{label}_{i}_{n_frame}.h5", "w")
                    f.create_dataset("hit", data=0)
                    f.create_dataset("id", data=file_name)
                    f.create_dataset("intensity", data=np.sum(corrected_data * mask))
                    f.create_dataset("refined_center", data=DetectorCenter)

                    now = datetime.now()
                    print(f"Current end time = {now}")


if __name__ == "__main__":
    main()
