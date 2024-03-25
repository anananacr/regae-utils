import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys

sys.path.append("../../utils/")
import utils


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Find peak positions according to peak range."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="path to the H5 data input file"
    )
    parser.add_argument("-p", "--peak_range", action="append", help="peak regions list")
    args = parser.parse_args(raw_args)

    peak_range = []
    list_of_peaks = []
    for idx, i in enumerate(args.peak_range):
        peak_range.append(int(i))
        if idx % 2 != 0:
            peak_range.sort()
            list_of_peaks.append(peak_range)
            peak_range = []
    f = h5py.File(args.input, "r+")
    rad_sub = [np.array(f["radial_x"]), np.array(f["radial"])]
    rad_sub = np.transpose(np.array(rad_sub))
    peak_px = []
    intensities = []

    for i in list_of_peaks:
        index = [
            np.where(rad_sub[:, 0] == i[0])[0][0],
            np.where(rad_sub[:, 0] == i[1])[0][0],
        ]
        rad_signal_cut = [
            rad_sub[:, 0][index[0] : index[1]],
            rad_sub[:, 1][index[0] : index[1]],
        ]

        peak_pos, half = utils.calc_fwhm(
            rad_signal_cut[1], -1, threshold=100, distance=4, height=0, width=2
        )
        print(peak_pos)
        while len(peak_pos) == 0:
            index[0] -= 1
            index[1] += 1
            rad_signal_cut = [
                rad_sub[:, 0][index[0] : index[1]],
                rad_sub[:, 1][index[0] : index[1]],
            ]
            peak_pos, half = utils.calc_fwhm(
                rad_signal_cut[1], -1, threshold=1, distance=4, height=0, width=2
            )

        if len(peak_pos) > 1:
            maxim = 0
            for k in peak_pos:
                if (rad_signal_cut[1][k]) > maxim:
                    max_peak = k
                    maxim = rad_signal_cut[1][k]
            peak_pos = [max_peak]

        peak_px.append(peak_pos + rad_signal_cut[0][0])

    print(peak_px)
    try:
        f.create_dataset("peak_position", data=peak_px)
        f.create_dataset("peak_range", data=np.array(list_of_peaks))
    except:
        del f["peak_position"]
        del f["peak_range"]
        f.create_dataset("peak_position", data=peak_px)
        f.create_dataset("peak_range", data=np.array(list_of_peaks))

    f.close()


if __name__ == "__main__":
    main()
