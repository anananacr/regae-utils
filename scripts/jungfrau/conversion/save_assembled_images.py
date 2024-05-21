import h5py
import argparse
import numpy as np
import om.lib.geometry as geometry
import os
import subprocess as sub
from PIL import Image
import fabio
import pathlib


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Assemble multi-panels H5 images according to the geometry file."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )

    parser.add_argument(
        "-g", "--geom", type=str, action="store", help="crystfel geometry file"
    )
    parser.add_argument(
        "-m", "--mask", type=str, action="store", help="mask file 0 bad pixels 1 good pixels"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output path"
    )
    args = parser.parse_args(raw_args)

    geometry_txt = open(args.geom, "r").readlines()

    # Open mask
    with h5py.File(f"{args.mask}", "r") as f:
        mask = np.array(f["/data/data"])


    geom_info = geometry.GeometryInformation(
            geometry_description=geometry_txt, geometry_format="crystfel"
        )
    pixel_maps = geom_info.get_pixel_maps()
    visual_img_shape = geometry._compute_min_array_shape(pixel_maps=pixel_maps)
    data_visualize = geometry.DataVisualizer(pixel_maps=pixel_maps)

    raw_folder = os.path.dirname(args.input)
    output_folder = args.output
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    cmd = f"cp {raw_folder}/info.txt {output_folder}"
    sub.call(cmd, shell=True)

    f = h5py.File(f"{args.input}_master.h5", "r")
    size = len(f["/entry/data/data"])
    label = (args.input).split("/")[-1]

    for i in range(size):
        try:
            data = np.array(f["/entry/data/data"][i])
            data[np.where(data <= 0)] = -1
        except OSError:
            print("skipped", i)
            continue
        visual_data = np.zeros(
            (visual_img_shape[0], visual_img_shape[1]), dtype=np.int32
        )
        visual_data =  data_visualize.visualize_data(data=data*mask)
        visual_data[np.where(visual_data<= 0)] = -1
        output_filename=f"{args.output}/{label}_{i:06}.tif"
        Image.fromarray(visual_data).save(f"{output_filename}")

    f.close()


if __name__ == "__main__":
    main()
