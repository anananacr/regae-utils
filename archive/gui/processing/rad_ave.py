# -*- coding: utf-8 -*-
import om.utils.crystfel_geometry as crystfel_geometry
import h5py
import numpy as np
import argparse
import skued


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Perform azimuthal integration of images from CrystFEL geometry file."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="path to the H5 data input file"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output H5 file"
    )

    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:
        data = np.array(f["sum_centered"])
    center = [int(data.shape[1] / 2), int(data.shape[0] / 2)]
    x, y = skued.azimuthal_average(
        data, center=center, angular_bounds=(0, 360), trim=True
    )

    f = h5py.File(args.output + ".h5", "a")
    f.create_dataset("radial_x", data=x)
    f.create_dataset("radial", data=y)
    f.close()


if __name__ == "__main__":
    main()
