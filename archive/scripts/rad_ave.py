# -*- coding: utf-8 -*-

import crystfel_geometry
import time
import h5py
import numpy as np
import argparse
from typing import Any, BinaryIO, List
import matplotlib.pyplot as plt
from PIL import Image
import center_finding
from utils import sum_nth, radial_average, sum_nth_mask
import utils
from multiprocessing import Pool
from powder import azimuthal_average
from skued import gaussian
from skued import baseline_dwt,baseline_dt
import peakutils

dark = None
gain = None

def main(raw_args=None):
    global dark, gain

    # argument parser
    parser = argparse.ArgumentParser(
        description="Decode and sum up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-m", "--mask", type=str, action="store",
        help="path to the virtual H5 mask file")
    parser.add_argument("-g", "--geom", type=str, action="store",
        help="path to the virtual H5 mask file")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:
        
        x = f["data"].shape[0]
        y = f["data"].shape[1]

        data=np.array(f["data"])


    f= h5py.File(args.mask, "r")
    mask=np.array(f['data/data'])
    f.close()

    masked_data=data*mask

    geometry_filename=args.geom
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2
        * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min())))
        + 2
    )
    x_minimum: int = (
        2
        * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min())))
        + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    frame_data=crystfel_geometry.apply_geometry_to_data(masked_data,geometry)

    center=[_img_center_x,_img_center_y]
    x,y=azimuthal_average(frame_data, center=center,angular_bounds=(0,360),trim=True)

    f= h5py.File(args.output+'.h5', "a")
    f.create_dataset('frame_data',data=frame_data)
    f.create_dataset('radial_x', data=x)
    f.create_dataset('radial', data=y)
    f.close()

if __name__ == '__main__':
    main()
