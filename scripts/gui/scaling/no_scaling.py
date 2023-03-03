import time
import h5py
import numpy
import subprocess as sub
import argparse
from typing import Any, BinaryIO, List
import matplotlib.pyplot as plt
from PIL import Image
import center_finding
from utils import sum_nth, radial_average, sum_nth_mask
import utils
from multiprocessing import Pool
from powder import azimuthal_average

def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Decode and sum up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-n", "--norm_file", type=str, action="store",
        help="normalize input to norm_file")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args(raw_args)
    cmd =f'cp {args.input} {args.output}.h5' 
    sub.call(cmd, shell=True)

if __name__ == '__main__':
    main()
