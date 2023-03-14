import sys

sys.path.append("../../utils/")
import h5py
import numpy as np
import argparse
import utils


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Scales image according to the low-resolution scattering signal."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to the H5 data master file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output H5 file"
    )

    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:
        norm_factor = np.mean(np.array(f["radial"])[:14])
        # print(norm_factor)
        radial = np.array(f["radial"])
        norm_rad = radial / (norm_factor)

        rad_signal_cut = []
        beam_cut = 0
        rad_signal_cut.append(np.array(f["radial_x"])[beam_cut:])
        rad_signal_cut.append(norm_rad[beam_cut:])

    baseline = utils.baseline_als(rad_signal_cut[1], 1e4, 0.1)
    rad_sub = rad_signal_cut[1] - baseline
    rad_sub[np.where(rad_sub < 0)] = 0

    g = h5py.File(args.input, "r")
    f = h5py.File(args.output + ".h5", "w")
    f.create_dataset("radial", data=rad_signal_cut[1])
    f.create_dataset("radial_x", data=rad_signal_cut[0])
    f.create_dataset("rad_sub", data=rad_sub)
    f.create_dataset("peak_position", data=np.array(g["peak_position"]))
    f.create_dataset("peak_range", data=np.array(g["peak_range"]))
    f.close()
    g.close()


if __name__ == "__main__":
    main()
