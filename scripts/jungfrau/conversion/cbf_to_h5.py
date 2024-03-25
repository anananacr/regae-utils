from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
import argparse
import numpy as np
import os
from PIL import Image
import h5py
import cbf
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

    input_file = fabio.open(args.input)
    data= input_file.data

    images_list=[]
    output_filename = f"{args.output}/{os.path.basename(args.input).split('.')[0]}.h5"
    with h5py.File(f"{output_filename}", "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"]="NXentry"
        grp_data = entry.create_group("data")
        grp_data.attrs["NX_class"]="NXdata"
        grp_data.create_dataset("data", data=np.array(data).astype(np.int32), compression="gzip")

if __name__ == "__main__":
    main()
