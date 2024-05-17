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
        description="Convert H5 to cbf. Deprecated CBFlib incompatibility with Python 3.10"
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
    content={
    
    }
    for i in range(0, len(paths[:])):
        file_name = paths[i][:-1]
        output_filename = f"{args.output}/{os.path.basename(file_name).split('.')[0]}.cbf"
        data = np.array(Image.open(file_name))
        data[np.where(data<0)]=0

        output=fabio.cbfimage.CbfImage(data=data)
        output.write(output_filename)

if __name__ == "__main__":
    main()
