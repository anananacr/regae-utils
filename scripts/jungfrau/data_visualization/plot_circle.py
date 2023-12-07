import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import resolution_rings
import argparse


def main():
    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot resolution rings. Parameters need to be correctly set in code."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    args = parser.parse_args()

    file_path = args.input
    hf = h5py.File(file_path, "r")

    ##HDF5_path
    data_name = "sum_frames_mask"
    # data_name='centered_data'
    data = np.array(hf[data_name])
    hf.close()
    ## Rings radius in pixels
    rings = [2, 40, 64, 98, 110]
    ## Center of the image [xc,yc]
    center = [515, 532]
    resolution_rings(data, center, rings, Imax=1e5)


if __name__ == "__main__":
    main()
