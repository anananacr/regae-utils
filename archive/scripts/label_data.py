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
import om.utils.crystfel_geometry as crystfel_geometry

central_signal=250


def apply_geom(data: np.ndarray, geometry_filename: str) -> np.ndarray:
    ## Apply crystfel geomtry file .geom
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2 * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min()))) + 2
    )
    x_minimum: int = (
        2 * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min()))) + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    corr_data = crystfel_geometry.apply_geometry_to_data(data, geometry)
    return corr_data



def label_intensity(image, sigma=0.1, min_value=100, max_value=200):

    # denoise the image with a Gaussian filters
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)
    # mask the image according to threshold
    binary_mask=np.ones(image.shape, dtype=bool)
    binary_mask[np.where(blurred_image<min_value)] = False
    binary_mask[np.where(blurred_image>max_value)] = False

    print(type(binary_mask[0,0]), binary_mask[0,0])
    # perform connected component analysis
    labeled_image = binary_mask.astype(int)
    return labeled_image

def connected_components(image, sigma=0.1, t=2000, background=100,connectivity=2):

    # denoise the image with a Gaussian filters
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)
    # mask the image according to threshold
    
    binary_mask = blurred_image < t

    print(type(binary_mask[0,0]), binary_mask[0,0])
    # perform connected component analysis
    labeled_image = skimage.measure.label(
        binary_mask, background=background, connectivity=connectivity, return_num=False
    )
    return labeled_image


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
        "-m",
        "--mask",
        type=str,
        action="store",
        help="path to the H5 mask file",
    )
    parser.add_argument(
        "-g",
        "--geom",
        type=str,
        action="store",
        help="geom file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output image"
    )

    args = parser.parse_args(raw_args)

    
    data = np.array(Image.open(args.input))
    
    with h5py.File(args.mask, "r") as f:
        mask = np.array(f["data/data"])
    mask=apply_geom(mask, args.geom)
    fig, (ax1, ax2),  = plt.subplots(1, 2, figsize=(8, 8))

    ax1.imshow(data*mask, cmap="jet", vmax=200)

    
    #labeled = connected_components(data*mask, sigma=0.1, t=10, background=100, connectivity=2)

    ## vaccum tube
    labeled = label_intensity(data*mask, sigma=1, min_value=10, max_value=1000)
    roi_tube=labeled.copy()
    

    ## rings and center
    labeled = label_intensity(data*mask, sigma=0.1, min_value=160, max_value=1000)
    roi_rings_and_center=labeled.copy()


    ## rings and center
    labeled = label_intensity(data*mask, sigma=1, min_value=160, max_value=170)
    roi=labeled.copy()

    #roi[np.where(labeled==1)]=0
    #roi[np.where(labeled==0)]=1
    #roi[np.where(mask==0)]=0
    
    #labeled_inverted=labeled.copy()
    #labeled_inverted[np.where(labeled==1)]=0
    #labeled_inverted[np.where(labeled==0)]=1

    ax2.imshow(roi*data, cmap="nipy_spectral")
    plt.show()


if __name__ == "__main__":
    main()
