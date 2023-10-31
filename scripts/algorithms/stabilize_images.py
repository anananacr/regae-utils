#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import get_format, gaussian, shift_image_by_n_pixels
import h5py
import math
from scipy.optimize import curve_fit
from PIL import Image
import os
from find_center_friedel import apply_geom

DetectorCenter = [587, 533]


def main():
    parser = argparse.ArgumentParser(description="Plot calculated center distribution.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        action="store",
        help="mask file",
    )
    parser.add_argument(
        "-g",
        "--geom",
        type=str,
        action="store",
        help="geometry file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    file_format = get_format(args.input)
    output_folder = args.output

    if file_format == "lst":
        for i in paths:
            f = h5py.File(f"{i[:-1]}", "r")
            center = np.array(f["refined_center"])
            image_id = str(np.array(f["id"]))
            error = math.sqrt(
                (center[0] - DetectorCenter[0]) ** 2
                + (center[1] - DetectorCenter[1]) ** 2
            )
            if error > 10:
                print(i[:-1])
            f.close()

            image_file = image_id[2:-1]
            assembled_path = os.path.dirname(image_file)
            basename = os.path.basename(image_file)
            index = int(basename.split(".")[0].split("_")[-1])
            folders = assembled_path.split("/")
            converted_path = ""
            for i in folders:
                if i == "assembled":
                    converted_path += "/converted"
                else:
                    converted_path += "/" + i
            converted_path = converted_path[1:]
            ## Apply static mask
            h5_file = h5py.File(f"{args.mask}", "r")
            mask = np.array(h5_file["data/data"])
            h5_file.close()

            ## Shift calculation
            assembled_data = np.array(Image.open(assembled_path + "/" + basename))
            h, w = assembled_data.shape
            shift = [int(np.round(w / 2 - center[0])), int(np.round(h / 2 - center[1]))]
            print(basename, center, w / 2, h / 2, shift)
            h5_file = h5py.File(f"{converted_path}/{basename[:-11]}_master.h5", "r")
            data = np.array(h5_file["data"][index])
            masked_data = data * mask
            h5_file.close()

            ## Apply shift
            shifted_data = shift_image_by_n_pixels(
                shift_image_by_n_pixels(data, shift[0], 1), shift[1], 0
            )
            shifted_data[np.where(shifted_data == 0)] = -1
            corrected_data = apply_geom(shifted_data, args.geom)

            pads_index = list(
                zip(np.where(corrected_data == 0)[0], np.where(corrected_data == 0)[1])
            )
            value = []
            for i in pads_index:
                try:
                    a = assembled_data[i[0] - shift[1], i[1] - shift[0]]
                    # print(a)
                    value.append([i, a])
                except IndexError:
                    value.append([i, -1])

            fig = plt.figure(figsize=(20, 5))
            ax = fig.add_subplot(121, title="Corrected data")
            ax.imshow(corrected_data, vmin=-1, vmax=100)
            ax = fig.add_subplot(122, title="Shifted data")
            ax.imshow(shifted_data, vmin=-1, vmax=100)
            # plt.show()
            plt.close("all")

            shifted_data = shift_image_by_n_pixels(
                shift_image_by_n_pixels(masked_data, shift[0], 1), shift[1], 0
            )
            shifted_data[np.where(shifted_data == 0)] = -1
            corrected_data = apply_geom(shifted_data, args.geom)

            for i in value:
                index, v = i
                corrected_data[index] = v

            fig = plt.figure(figsize=(20, 5))
            ax = fig.add_subplot(121, title="Corrected data")
            ax.imshow(corrected_data, vmin=-1, vmax=100)
            ax = fig.add_subplot(122, title="Shifted data")
            ax.imshow(shifted_data, vmin=-1, vmax=100)
            # plt.show()
            plt.close("all")

            corrected_mask = apply_geom(mask, args.geom)
            shifted_mask = shift_image_by_n_pixels(
                shift_image_by_n_pixels(corrected_mask, shift[0], 1), shift[1], 0
            )
            plt.imshow(shifted_mask)
            # plt.show()
            plt.close("all")

            h, w = corrected_data.shape
            # print(h,w)
            plt.imshow(corrected_data * shifted_mask, vmin=-1, vmax=100)
            plt.scatter(w / 2, h / 2, c="r")
            # plt.show()
            plt.close("all")

            final_data = corrected_data * shifted_mask
            final_data[np.where(final_data == 0)] = -1
            Image.fromarray(final_data).save(f"{args.output}/{basename}")


if __name__ == "__main__":
    main()
