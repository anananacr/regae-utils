#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import get_format, gaussian
import h5py
import math
from scipy.optimize import curve_fit

DetectorCenter = [606, 539]
frequency = 12.5
frames_per_step = 100


def calculate_time_point_from_path(file_path: str, frame: int):
    # print(((file_path.split('/')[-1]).split('.')[0]).split('_')[-1])
    file_index = int(((file_path.split("/")[-1]).split(".")[0]).split("_")[-1])
    n = (frames_per_step * file_index) + frame
    return n / frequency


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
        "-l",
        "--label",
        type=str,
        action="store",
        help="path to list of data files .lst",
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
    label = "center_distribution_" + args.label
    g = open(f"{args.output}/plots/beam_center.csv", "w")
    g.write("time center_x center_y\n")
    # print(label)
    center_x = []
    center_y = []
    x_min = DetectorCenter[0] - 10
    x_max = DetectorCenter[0] + 10
    y_min = DetectorCenter[1] - 10
    y_max = DetectorCenter[1] + 10
    time = []
    counts = []
    if file_format == "lst":
        for i in paths[:]:
            print(i)
            try:
                f = h5py.File(f"{i[:-1]}", "r")
                center = np.array(f["data/refined_center"])
                intensity = np.array(f["data/intensity"])
                file_path = str(np.array(f["data/id"]))
                # frame=int(np.array(f["index"]))
                frame = 0
                error = math.sqrt(
                    (center[0] - DetectorCenter[0]) ** 2
                    + (center[1] - DetectorCenter[1]) ** 2
                )
                if (
                    center[1] > y_min
                    and center[1] < y_max
                    and center[0] < x_max
                    and center[0] > x_min
                ):
                    timestamp = calculate_time_point_from_path(file_path, frame)
                    time.append(timestamp)
                    center_x.append(center[0])
                    center_y.append(center[1])
                    counts.append(intensity)
                    g.write(f"{timestamp} {center[0]} {center[1]}\n")
                f.close()
            # if error>10:
            #    print(i[:-1])
            # f.close()
            except:
                continue
                print(i[:-1])
            # except:
            #    print("OS", i[:-1])
    # print(len(center_x))
    g.close()

    norm_intensity = counts / np.median(counts)

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(311, title="Direct beam @REGAE")

    ax.set_ylabel("Detector center in x (pixel)", fontsize=10)
    ax.set_xlabel("Time (s)")
    # ax.set_xlim(0,8000)
    ax.scatter(time, center_x, marker=".", s=2)

    ax = fig.add_subplot(312)

    ax.set_ylabel("Detector center in y (pixel)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.scatter(time, center_y, color="orange", marker=".", s=2)
    # ax.set_xlim(0,8000)

    ax = fig.add_subplot(313)

    ax.set_ylabel("Normalized intensity", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.scatter(time, norm_intensity, color="green", marker=".", s=2)
    # ax.set_xlim(0,8000)
    ax.legend()
    plt.savefig(f"{args.output}/plots/{label}_time.png")
    plt.show()


if __name__ == "__main__":
    main()
