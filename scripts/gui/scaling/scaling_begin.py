import time
import h5py
import numpy
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
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args(raw_args)


    with h5py.File(args.input, "r") as f:
         norm_factor=numpy.mean(numpy.array(f['rad_average_mask'])[:14])
         #print(norm_factor)
         data=numpy.array(f['sum_frames_mask'])
         norm_data=data/(norm_factor)
         norm_rad=1000*numpy.array(f['rad_average_mask'])/(norm_factor)
    
         rad_signal_cut=[]
         beam_cut=0
         rad_signal_cut.append(numpy.array(f['rad_x'])[beam_cut:])
         rad_signal_cut.append(norm_rad[beam_cut:])

    baseline=utils.baseline_als(rad_signal_cut[1],1e4, 0.1)
    rad_sub=rad_signal_cut[1]-baseline
    rad_sub[numpy.where(rad_sub<0)]=0

    peaks, half=utils.calc_fwhm(rad_sub,6,threshold=0,distance=15, height=0, width=8)
    peak_px=peaks+rad_signal_cut[0][0]
    fwhm_over_rad=[]
    intensity=[]
    for i in range(len(peaks)):
        fwhm_over_rad.append(half[0][i]/peak_px[i])
        intensity.append(rad_sub[peaks[i]])

    f= h5py.File(args.output+'.h5', "w")
    f.create_dataset('rad_sub',data=rad_sub)
    f.create_dataset('sum_frames_mask',data=data)
    f.create_dataset('norm_sum_frames',data=norm_data)
    f.create_dataset('rad_average_mask',data=rad_signal_cut[1])
    f.create_dataset('rad_x',data=rad_signal_cut[0])
    f.create_dataset('intensity',data=intensity)
    f.create_dataset('fwhm_radius',data=fwhm_over_rad)
    f.create_dataset('fwhm',data=half[0])
    f.create_dataset('peak_position',data=peak_px)
    f.close()

if __name__ == '__main__':
    main()
