# -*- coding: utf-8 -*-
import sys
sys.path.append("../../utils")
sys.path.append("../../jungfrau/conversion")
import om.utils.crystfel_geometry as crystfel_geometry
import time
import h5py
import numpy as np
import argparse
from convert_images import apply_calibration, filter_data
from typing import Any, BinaryIO, List
import matplotlib.pyplot as plt
import center_finding
from multiprocessing import Pool
import peakutils
from utils import sum_nth, radial_average, sum_nth_mask, baseline_als
import skued 


dark = None
gain = None

def main(raw_args=None):
    global dark, gain

    # argument parser
    parser = argparse.ArgumentParser(
        description="Decode and sum up Jungfrau images no centering routine.")
    parser.add_argument("-p1", "--pedestal1", type=str, action="store",
        help="path to the pedestal file for module 1")
    parser.add_argument("-p2", "--pedestal2", type=str, action="store",
        help="path to the pedestal file for module 2")
    parser.add_argument("-g1", "--gain1", type=str, action="store",
        help="path to the gain info file for module 1")
    parser.add_argument("-g2", "--gain2", type=str, action="store",
        help="path to the gain info file for module 2")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-m", "--mask", type=str, action="store",
        help="path to the virtual H5 mask file")
    parser.add_argument("-b", "--begin_frame", type=int, action="store",
        help="process frames starting from b ")
    parser.add_argument("-n", "--n_frames", type=int, action="store",
        help="number of frames to accumulate")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output hdf5 file")

    args = parser.parse_args(raw_args)

    # Check arguments
    if not (args.pedestal1 and args.pedestal2
            and args.gain1 and args.gain2
            and args.input and args.output):
        print("missing argument")
        return

    # Import dark and gain files
    num_panels: int = 2
    dark_filenames = [args.pedestal1, args.pedestal2]
    gain_filenames = [args.gain1, args.gain2]
    dark = np.ndarray(
        (3, 512 * num_panels, 1024), dtype=np.float32)
    gain = np.ndarray(
        (3, 512 * num_panels, 1024), dtype=np.float64)
    panel_id: int
    for panel_id in range(num_panels):
        gain_file: BinaryIO = open(gain_filenames[panel_id], "rb")
        dark_file: Any = h5py.File(dark_filenames[panel_id], "r")
        gain_mode: int
        for gain_mode in range(3):
            dark[gain_mode, 512 * panel_id : 512 * (panel_id + 1), :] = dark_file[
                "gain%d" % gain_mode][:]
            gain[gain_mode, 512 * panel_id : 512 * (panel_id + 1), :] = np.fromfile(
                    gain_file,
                    dtype=np.float64,
                    count=1024 * 512
                ).reshape(512, 1024)
        gain_file.close()
        dark_file.close()

    #Loop through series
    with h5py.File(args.input, "r") as f:
        #n_frames = f["data"].shape[0]
        n_frames =args.n_frames
        begin= args.begin_frame
        x = f["entry/data/data"].shape[1]
        y = f["entry/data/data"].shape[2]

        #print('%s frames %sx%s' % (n_frames, x, y))
        raw_data=f["entry/data/data"][begin:begin+n_frames]
        filt_data=raw_data.copy()
        for i in range(n_frames):
             skip=filter_data(filt_data[i])
             if skip==1:
                 filt_data[i]=dark[0]

        ## Uncomment to save float numbers
        #acc_image = np.zeros((n_frames, y, x), dtype=np.float64)
        
        acc_image = np.zeros((n_frames, y, x), dtype=np.int32)
        #print('Applying calibration')

        d=np.zeros((n_frames, y, x), dtype=np.int32)
        for idx, i in enumerate(filt_data):
             d[idx]=apply_calibration(i)

        for idx, i in enumerate(d):
            acc_image[idx] = i.reshape(y, x)

    ctr_img_lst=[]
    data=acc_image[:n_frames]
    #f= h5py.File(args.mask, "r")
    #mask=center_finding.include_gaps_np(np.array(f['data/data']))
    #f.close()
    #sum_img=sum_nth(np.array(data),1,0)
    
    #center=center_finding.calc_com_iter_mask(sum_img,mask)
    #print(center)
    #center=[578,515]
    #acc_corr_image=[center_finding.include_gaps_np(x) for x in data]
    
    ctr_img_lst=[]

    #for idx,i in enumerate(acc_corr_image):
    #    masked_data=i*mask
    #    new_data, shift=center_finding.bring_center_to_point(masked_data, center)
    #    ctr_img_lst.append(new_data)
    #ctr_img_lst.append(new_data)

    #print(f'Azimuthal integration')

    #masked_data_final=sum_nth(np.array(ctr_img_lst),1,0)

    #rad_signal_masked=radial_average(masked_data_final, None)
    #rad_signal_masked=skued.azimuthal_average(masked_data_final, center=(515,532),angular_bounds=(0,360),trim=True)
    #beam_cut=0
    #rad_signal_masked[0]=rad_signal_masked[0][beam_cut:]
    #rad_signal_masked[1]=rad_signal_masked[1][beam_cut:]
    #baseline=baseline_als(rad_signal_masked[1],1e5, 0.1)
    #mask_center=np.zeros(len(rad_signal_masked[0])+1, dtype=bool)



    #base_dwt = skued.baseline_dwt(rad_signal_masked[1][beam_cut:], level=5,max_iter = 150, wavelet = 'sym6',background_regions=rad_signal_masked[0][beam_cut:350])
    #rad_sub=np.transpose(np.array([rad_signal_masked[0][beam_cut:],rad_signal_masked[1][beam_cut:]-base_dwt]))
    #base_dt=skued.baseline_dt(rad_signal_masked[1], level=6,wavelet = 'qshift3', max_iter = 150, background_regions=rad_signal_masked[0][0:350])
    #base_pu=peakutils.baseline(rad_signal_masked[1])
    #print(args.input)
    #plt.plot(rad_signal_masked[0], rad_signal_masked[1],label='data')
    #plt.plot(rad_signal_masked[0],baseline,label='als')
    #plt.plot(rad_signal_masked[0],base_dwt,label='dwt')
    #plt.plot(rad_signal_masked[0],base_dt,label='dt')
    #plt.plot(rad_signal_masked[0],base_dt,label='peak_utils')
    #rad_sub=rad_signal_masked
    
    #plt.legend()
    #plt.show()
    #find peaks

    #write output
    f= h5py.File(args.output+'.h5', "w")
    #f.create_dataset('data',data=sum_img)
    f.create_dataset('data',data=acc_image)
    #f.create_dataset('centered_data',data=ctr_img_lst)
    #f.create_dataset('sum_frames',data=sum_img)
    #f.create_dataset('rad_sub',data=rad_sub[:,1])
    #f.create_dataset('rad_sub',data=rad_sub[1])
    #f.create_dataset('sum_frames_mask',data=masked_data_final)
    #f.create_dataset('rad_average_mask',data=rad_signal_masked[1][beam_cut:])
    #f.create_dataset('rad_x',data=rad_sub[:,0])
    #f.create_dataset('rad_x',data=rad_sub[0])


    f.close()

if __name__ == '__main__':
    main()
