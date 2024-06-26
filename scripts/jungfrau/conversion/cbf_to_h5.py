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
        description="Convert cbf files to H5. Deprecated incompatibility of Python 3.10 to CBFlib."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="input h5 file",
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
