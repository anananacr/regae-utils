#!/usr/bin/env python3.7

import h5py
import argparse
import numpy as np
import om.utils.crystfel_geometry as crystfel_geometry
import cbf

def apply_geom(data:np.ndarray, geometry_filename: str)-> np.ndarray:
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
    return corr_data.astype(np.int32)

def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Convert JUNGFRAU 1M H5 images collected at REGAE for rotational data step/fly scan and return images in rotation sequence according tro the file index."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )

    parser.add_argument(
        "-g", "--geom", type=str, action="store", help="crystfel geometry file"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="hdf5 output path"
    )
    args = parser.parse_args(raw_args)


    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(args.geom)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2 * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min()))) + 2
    )
    x_minimum: int = (
        2 * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min()))) + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    f = h5py.File(f"{args.input}_master.h5", "r")
    size = len(f["data"])
    idy=9
    
    label=(args.input).split('/')[-1]

    for i in range(size):       
        try:
            raw = np.array(f["data"][i])
        except OSError:
            print("skipped", i)
            continue
        corr_frame = np.zeros((visual_img_shape[0],visual_img_shape[1]), dtype=np.int32)
        corr_frame = apply_geom(raw, args.geom)
        cbf.write(f'{args.output}/{label}_{i:06}.cbf', corr_frame)

    f.close()

if __name__ == "__main__":
    main()

    