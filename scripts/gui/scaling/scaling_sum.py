import h5py
import numpy
import argparse
import matplotlib.pyplot as plt
import skued


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Scales image according to the total intensity signal deposited in the normalization file."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to the H5 data master file",
    )
    parser.add_argument(
        "-n",
        "--norm_file",
        type=str,
        action="store",
        help="normalize input to norm_file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output image"
    )

    args = parser.parse_args(raw_args)

    with h5py.File(args.norm_file, "r") as f:
        norm_file_sum = numpy.sum(numpy.array(f["sum_centered"]))

    with h5py.File(args.input, "r") as f:
        data_sum = numpy.sum(numpy.array(f["sum_centered"]))
        norm_factor = data_sum / norm_file_sum
        norm_data = numpy.array(f["sum_centered"]) / (norm_factor)

    _img_center_x: int = int(norm_data.shape[1] / 2)
    _img_center_y: int = int(norm_data.shape[0] / 2)
    center = [_img_center_x, _img_center_y]

    rad_signal_masked = skued.azimuthal_average(
        norm_data, center=center, angular_bounds=(0, 360), trim=True
    )
    base_dwt = skued.baseline_dwt(
        rad_signal_masked[1],
        level=6,
        max_iter=150,
        wavelet="sym6",
        background_regions=rad_signal_masked[0][0:350],
    )
    rad_sub = numpy.transpose(
        numpy.array([rad_signal_masked[0], rad_signal_masked[1] - base_dwt])
    )

    # write output
    g = h5py.File(args.input, "r")
    f = h5py.File(args.output + ".h5", "w")
    f.create_dataset("radial", data=rad_signal_masked[1])
    f.create_dataset("radial_x", data=rad_sub[:, 0])
    f.create_dataset("rad_sub", data=rad_sub[:, 1])
    f.create_dataset("peak_position", data=numpy.array(g["peak_position"]))
    f.create_dataset("peak_range", data=numpy.array(g["peak_range"]))
    f.close()
    g.close()


if __name__ == "__main__":
    main()
