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

DetectorCenter = [606.44, 539]
max_deviation = 1000


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
            center = np.array(f["data/refined_center"])
            image_id = str(np.array(f["data/id"]))
            distance = math.sqrt(
                (center[0] - DetectorCenter[0]) ** 2
                + (center[1] - DetectorCenter[1]) ** 2
            )
            if distance > max_deviation:
                print(
                    f"Warning!! Refined center more than {max_deviation} pixels far from the median for file {image_id[2:-1]}"
                )
                # center = DetectorCenter
            f.close()

            image_file = image_id[2:-1]
            basename = os.path.basename(image_file)


            ## Shift calculation
            assembled_data = np.array(Image.open(image_file))
            h, w = assembled_data.shape
            shift = [int(np.round(w / 2 - center[0])), int(np.round(h / 2 - center[1]))]

            ## Apply shift
            shifted_data = shift_image_by_n_pixels(
                shift_image_by_n_pixels(assembled_data, shift[0], 1), shift[1], 0
            )

            final_data = shifted_data
            final_data[np.where(final_data == 0)] = -1
            final_data = np.array(final_data, dtype=np.int32)
            Image.fromarray(final_data).save(f"{args.output}/{basename}")


if __name__ == "__main__":
    main()
