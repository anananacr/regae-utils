from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
import argparse
import numpy as np
import os
from PIL import Image
import h5py

def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of diffraction patterns fro MHz beam sweeping serial crystallography."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    images_list=[]
    for i in range(0, len(paths[:])):
        file_name = paths[i][:-1]
        output_filename = f"{args.output}/{os.path.basename(file_name).split('.')[0]}.h5"
        data = np.array(Image.open(file_name))
        data[np.where(data<0)]=0
        images_list.append(data)

    with h5py.File(f"{output_filename}", "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"]="NXentry"
        grp_data = entry.create_group("data")
        grp_data.attrs["NX_class"]="NXdata"
        grp_data.create_dataset("data", data=np.array(images_list).astype(np.int32), compression="gzip")
        

if __name__ == "__main__":
    main()
