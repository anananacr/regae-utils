import sys

sys.path.append("../scaling")
import scaling_begin
import scaling_sum
import no_scaling

sys.path.append("../merge")
import merge_scan
import merge_average_radial

sys.path.append("../fit")
import fit_peaks_gaus_lin as fit_peaks

import scan_radial_to_file
import select_peak

import numpy as np
import h5py
import argparse
import subprocess as sub
import glob
from datetime import datetime


def drawProgressBar(percent, barLen=20):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write(
        "Data processing status [{:<{}}] {:.2f}%".format(
            "=" * int(barLen * percent), barLen, percent * 100
        )
    )
    sys.stdout.flush()


def run(peak_args, raw_args):
    """
    Scale data, fit peak positions and merge scans.
    Second step of data processing from select_peaks.py GUI.

    Parameters
    ----------
    peak_args: List
        Input strings from select peaks GUI.
    raw_args: List
        Input strings from GUI.
    Returns
    ----------
    bool
        Processing status, True when complete.
    """

    PEDAL_DATE = raw_args[0]
    DATE = raw_args[1]
    INPUT = peak_args[3][:]
    LABEL = raw_args[3]
    TOTAL = int(raw_args[4])
    STEP = int(raw_args[5])
    ZERO_DELAY = float(raw_args[6])
    FIRST_INDEX = int(raw_args[7])
    FIRST_DELAY = float(raw_args[8])
    LAST_INDEX = int(raw_args[9])
    LAST_DELAY = float(raw_args[10])
    DELAY_STEP = float(raw_args[11])
    N_PEAKS = int(peak_args[1])
    SAME_RANGE = peak_args[5]
    PEAKS_RANGE = peak_args[6:]
    scan_mode = int(raw_args[12])

    # DEFINE HERE YOUR DATA ROOT
    ROOT_FILES = INPUT[0][:44] + "scratch_cc/rodria/processed"

    NORM = f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{FIRST_INDEX}_0.h5"
    CMD = f"mkdir {ROOT_FILES};"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)
    CMD = f"mkdir {ROOT_FILES}/scaled/; mkdir {ROOT_FILES}/fit/; mkdir {ROOT_FILES}/merged/; mkdir {ROOT_FILES}/average/;"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    CMD = f"rm -r {ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}*; rm -r {ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}*; rm -r {ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}*; rm -r {ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}*;"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    CMD = f"mkdir {ROOT_FILES}/scaled/{DATE}/; mkdir {ROOT_FILES}/fit/{DATE}/;mkdir {ROOT_FILES}/merged/{DATE}/; mkdir {ROOT_FILES}/average/{DATE}/;"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time begin =", current_time)

    list_of_peaks = PEAKS_RANGE

    step = 1 / ((LAST_INDEX - FIRST_INDEX + 1) * (TOTAL / STEP))
    percent = 0
    n = 0
    part = int(TOTAL / STEP)

    for idx, i in enumerate(INPUT):
        args_list = ["-i", f"{i}"] + list_of_peaks[
            4 * int(N_PEAKS) * n : 4 * int(N_PEAKS) * (n + 1)
        ]
        try:
            select_peak.main(args_list)
        except:
            break
        if (idx + 1) % part == 0 and SAME_RANGE == False:
            n += 1

    percent = 0
    rejection_list = ""
    r_list = ""
    f = open(f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_report.txt", "w")
    f.close()

    for i in range(FIRST_INDEX, LAST_INDEX + 1, 1):

        for j in range(0, TOTAL, STEP):
            drawProgressBar(percent)

            args_list = [
                "-i",
                f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{i}_{j}.h5",
                "-n",
                f"{NORM}",
                "-o",
                f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_sum_{i}_{j}",
            ]
            scaling_sum.main(args_list)

            args_list = [
                "-i",
                f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{i}_{j}.h5",
                "-o",
                f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_beg_{i}_{j}",
            ]
            scaling_begin.main(args_list)

            args_list = [
                "-i",
                f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{i}_{j}.h5",
                "-o",
                f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_no_{i}_{j}",
            ]
            no_scaling.main(args_list)

            args_list = [
                "-i",
                f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_sum_{i}_{j}.h5",
                "-o",
                f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_sum_{i}_{j}",
            ]
            fit_peaks.main(args_list)

            args_list = [
                "-i",
                f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_beg_{i}_{j}.h5",
                "-o",
                f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_beg_{i}_{j}",
            ]
            fit_peaks.main(args_list)

            args_list = [
                "-i",
                f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_no_{i}_{j}.h5",
                "-o",
                f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_no_{i}_{j}",
            ]

            fit_peaks.main(args_list)

            percent += step

        args_list = [
            "-i",
            f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_sum_{i}_",
            "-o",
            f"{ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}_sum_{i}",
        ]
        merge_scan.main(args_list)

        args_list = [
            "-i",
            f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_beg_{i}_",
            "-o",
            f"{ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}_beg_{i}",
        ]
        merge_scan.main(args_list)

        args_list = [
            "-i",
            f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_no_{i}_",
            "-o",
            f"{ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}_no_{i}",
        ]
        merge_scan.main(args_list)

        args_list = [
            "-i",
            f"{ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}_sum_{i}",
            "-o",
            f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_sum_{i}",
        ]
        merge_average_radial.main(args_list)

        args_list = [
            "-i",
            f"{ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}_beg_{i}",
            "-o",
            f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_beg_{i}",
        ]
        merge_average_radial.main(args_list)

        args_list = [
            "-i",
            f"{ROOT_FILES}/merged/{DATE}/{DATE}_{LABEL}_no_{i}",
            "-o",
            f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_no_{i}",
        ]
        merge_average_radial.main(args_list)

        n_files_before_fit = len(
            glob.glob(f"{ROOT_FILES}/scaled/{DATE}/{DATE}_{LABEL}_no_{i}_*.h5")
        )
        n_files_after_fit = len(
            glob.glob(f"{ROOT_FILES}/fit/{DATE}/{DATE}_{LABEL}_no_{i}_*.h5")
        )

        path = f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_no_{i}.h5"
        hf = h5py.File(path, "r")
        data = np.array(hf["r_squared"])
        data_err = np.array(hf["r_squared_err"])
        hf.close()
        rejection_list += (
            f"[_{i}.h5 {round((1-(n_files_after_fit/n_files_before_fit))*100,2)}%]\n"
        )
        try:
            for idx, k in enumerate(data):
                r_list += f"[_{i}.h5 peak_{idx} {round(100*k,1)}±{round(100*data_err[idx],1)}%]\n"
        except TypeError:
            continue

    args_list = [
        "-i",
        f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_sum",
        "-fi",
        f"{FIRST_INDEX}",
        "-fp",
        f"{FIRST_DELAY}",
        "-li",
        f"{LAST_INDEX}",
        "-lp",
        f"{LAST_DELAY}",
        "-s",
        f"{DELAY_STEP}",
        "-t0",
        f"{ZERO_DELAY}",
        "-o",
        f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_sum_time_scans",
        "-r",
        "0",
        "-m",
        f"{scan_mode}",
    ]
    scan_radial_to_file.main(args_list)

    args_list = [
        "-i",
        f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_beg",
        "-fi",
        f"{FIRST_INDEX}",
        "-fp",
        f"{FIRST_DELAY}",
        "-li",
        f"{LAST_INDEX}",
        "-lp",
        f"{LAST_DELAY}",
        "-s",
        f"{DELAY_STEP}",
        "-t0",
        f"{ZERO_DELAY}",
        "-o",
        f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_beg_time_scans",
        "-r",
        "0",
        "-m",
        f"{scan_mode}",
    ]
    scan_radial_to_file.main(args_list)

    args_list = [
        "-i",
        f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_no",
        "-fi",
        f"{FIRST_INDEX}",
        "-fp",
        f"{FIRST_DELAY}",
        "-li",
        f"{LAST_INDEX}",
        "-lp",
        f"{LAST_DELAY}",
        "-s",
        f"{DELAY_STEP}",
        "-t0",
        f"{ZERO_DELAY}",
        "-o",
        f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_no_time_scans",
        "-r",
        "0",
        "-m",
        f"{scan_mode}",
    ]
    scan_radial_to_file.main(args_list)

    drawProgressBar(percent)

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("\nCurrent Time end =", current_time)

    print(f"\nFit rejection:\n{rejection_list}")
    print(f"\nMean R²:\n{r_list}")
    with open(f"{ROOT_FILES}/average/{DATE}/{DATE}_{LABEL}_report.txt", "+a") as f:
        f.write(
            f"\nCurrent Time end ={current_time}\nFit rejection:\n{rejection_list}\nMean R²:\n{r_list}"
        )
    return 1
