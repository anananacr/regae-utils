import time
import h5py
import numpy
import crystfel_geometry
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

    with h5py.File(args.norm_file, "r") as f:
         norm_file_sum=numpy.sum(numpy.array(f['frame_data']))

    with h5py.File(args.input, "r") as f:
         data_sum=numpy.sum(numpy.array(f['frame_data']))
         norm_factor=data_sum/norm_file_sum
         norm_data=numpy.array(f['frame_data'])/(norm_factor)

    _img_center_x: int = int(norm_data.shape[1] / 2)
    _img_center_y: int = int(norm_data.shape[0] / 2)
    center=[_img_center_x,_img_center_y]
    #print(norm_factor)
    rad_signal_masked=azimuthal_average(norm_data, center=center,angular_bounds=(0,360),trim=True)
    base_dwt = baseline_dwt(rad_signal_masked[1], level=6,max_iter = 150, wavelet = 'sym6',background_regions=rad_signal_masked[0][0:350])
    rad_sub=numpy.transpose(numpy.array([rad_signal_masked[0],rad_signal_masked[1]-base_dwt]))
    #write output


    g= h5py.File(args.input, "r")
    f= h5py.File(args.output+'.h5', "w")
    f.create_dataset('radial',data=rad_signal_masked[1])
    f.create_dataset('radial_x',data=rad_sub[:,0])
    f.create_dataset('rad_sub',data=rad_sub[:,1])
    f.create_dataset('peak_position',data=numpy.array(g['peak_position']))
    f.create_dataset('peak_range',data=numpy.array(g['peak_range']))
    f.close()

if __name__ == '__main__':
    main()
