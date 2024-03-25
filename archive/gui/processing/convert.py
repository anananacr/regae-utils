import numpy as np
import subprocess as sub

import sys

sys.path.append("../centering")
import sumup_center

import rad_ave

import multiprocessing
from datetime import datetime


def run(raw_args):
    """
    Convert RAW data from JUNGFRAU 1M and take the radial average profile from center given by geometry file.
    First step of data processing from regae_autoproc.py GUI.

    Parameters
    ----------
    raw_args: List
        Input strings from GUI.
    Returns
    ----------
    bool
        Processing status, True when complete.
    """
    PEDAL_DATE = raw_args[0]
    DATE = raw_args[1]
    INPUT = raw_args[2]
    LABEL = raw_args[3]
    TOTAL = int(raw_args[4])
    STEP = int(raw_args[5])
    ZERO_DELAY = float(raw_args[6])
    FIRST_INDEX = int(raw_args[7])
    FIRST_DELAY = float(raw_args[8])
    LAST_INDEX = int(raw_args[9])
    LAST_DELAY = float(raw_args[10])
    DELAY_STEP = float(raw_args[11])

    # DEFINE HERE YOUR DATA ROOT
    ROOT_FILES = INPUT[:44] + "scratch_cc/rodria/processed"
    print(f"Root files:{ROOT_FILES}")

    CMD = f"mkdir {ROOT_FILES};"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    # TODO check if pedestal is in folder
    CMD = f"cp -r {ROOT_FILES}/../../../processed/calib/* {ROOT_FILES}/calib;"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    # Prep data structure for autoproc
    CMD = f"mkdir {ROOT_FILES}/centered/; mkdir {ROOT_FILES}/scaled/; mkdir {ROOT_FILES}/fit/; mkdir {ROOT_FILES}/merged/; mkdir {ROOT_FILES}/average/;"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    CMD = f"mkdir {ROOT_FILES}/centered/{DATE}/;"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    CMD = f"rm -r {ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}*"
    sub.run(CMD, shell=True, stdout=sub.PIPE, universal_newlines=True)

    args_list_sum = []
    args_ave = []

    for i in range(FIRST_INDEX, LAST_INDEX + 1, 1):
        for j in range(0, TOTAL, STEP):
            args_list_sum.append(
                [
                    "-p1",
                    f"{ROOT_FILES}/calib/pedal_d0_{PEDAL_DATE}.h5",
                    "-p2",
                    f"{ROOT_FILES}/calib/pedal_d1_{PEDAL_DATE}.h5",
                    "-g1",
                    f"{ROOT_FILES}/calib/gainMaps_M283.bin",
                    "-g2",
                    f"{ROOT_FILES}/calib/gainMaps_M281.bin",
                    "-i",
                    f"{INPUT}_master_{i}.h5",
                    "-m",
                    f"{ROOT_FILES}/calib/mask_v0.h5",
                    "-g",
                    f"/home/rodria/proc/20221114/20221007_Au_{i}/JF_regae_v1_{i}.geom",
                    "-n",
                    f"{STEP}",
                    "-b",
                    f"{j}",
                    "-o",
                    f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{i}_{j}",
                ]
            )
            args_ave.append(
                [
                    "-i",
                    f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{i}_{j}.h5",
                    "-o",
                    f"{ROOT_FILES}/centered/{DATE}/{DATE}_{LABEL}_{i}_{j}",
                ]
            )

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time begin =", current_time)

    pool = multiprocessing.Pool(20)
    args = np.array(args_list_sum, dtype=object)
    with pool:
        pool.map(sumup_center.main, args)

    ave = 0
    for idx, i in enumerate(args_list_sum):
        print(idx)
        start = datetime.now()
        rad_ave.main(args_ave[idx])
        end = datetime.now()
        delta = end - start
        ave += 10 / float(delta.seconds + (delta.microseconds * 1e-6))

    ave = round((ave / (idx + 1)), 2)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time end images conversion=", current_time)
    return 1
