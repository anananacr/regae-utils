from scipy.ndimage.measurements import label
from skimage import filters
import numpy as np
import argparse
import h5py
import cv2
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
from PIL import Image


def connected_components(image, sigma=0.1, t=100, connectivity=2):

    # denoise the image with a Gaussian filters
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)
    # mask the image according to threshold
    binary_mask = blurred_image < t
    # perform connected component analysis
    labeled_image, count = skimage.measure.label(
        binary_mask, connectivity=connectivity, return_num=True
    )
    return labeled_image, count


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Connected components labelling of input HDF5 file."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to the H5 data master file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output image"
    )

    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:
        data = np.array(f["data"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    plt.imshow(data[80], cmap="jet", vmax=200)

    plt.show()
    for idx, i in enumerate(data[80:100]):

        labeled, count = connected_components(i)
        plt.imshow(labeled, cmap="nipy_spectral")
        plt.show()


if __name__ == "__main__":
    main()
