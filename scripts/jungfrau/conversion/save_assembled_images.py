import h5py
import argparse
import numpy as np
import om.lib.geometry as geometry
import os
import subprocess as sub
from PIL import Image
import fabio


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

    geometry_txt = open(args.geom, "r").readlines()
    geom_info = geometry.GeometryInformation(
            geometry_description=geometry_txt, geometry_format="crystfel"
        )
    pixel_maps = geom_info.get_pixel_maps()
    visual_img_shape = geometry._compute_min_array_shape(pixel_maps=pixel_maps)
    data_visualize = geometry.DataVisualizer(pixel_maps=pixel_maps)

    raw_folder = os.path.dirname(args.input)
    output_folder = args.output
    cmd = f"cp {raw_folder}/info.txt {output_folder}"
    sub.call(cmd, shell=True)

    f = h5py.File(f"{args.input}_master.h5", "r")
    size = len(f["data"])

    label = (args.input).split("/")[-1]

    for i in range(size):
        try:
            data = np.array(f["data"][i])
            data[np.where(data <= 0)] = -1
        except OSError:
            print("skipped", i)
            continue
        visual_data = np.zeros(
            (visual_img_shape[0], visual_img_shape[1]), dtype=np.int32
        )
        visual_data =  data_visualize.visualize_data(data=data)
        visual_data[np.where(visual_data<= 0)] = -1
        #output_filename=f"{args.output}/{label}_{i:06}.cbf"
        #output=fabio.cbfimage.CbfImage(data=visual_data)
        #output.write(output_filename)
        # cbf.write(f'{args.output}/{label}_{i:06}.cbf', visual_data)
        Image.fromarray(visual_data).save(f"{args.output}/{label[:-7]}_{i:06}.tif")

    f.close()


if __name__ == "__main__":
    main()
