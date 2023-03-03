# -*- coding: utf-8 -*-

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

dark = None
gain = None


def apply_calibration(data: numpy.ndarray) -> numpy.ndarray:
    """
    Applies the calibration to a detector data frame.

    This function determines the gain stage of each pixel in the provided data
    frame, and applies the relevant gain and offset corrections.

    Arguments:

        data: The detector data frame to calibrate.

    Returns:

        The corrected data frame.
    """
    corrected_data: numpy.ndarray = data.astype(numpy.float32)
    
    where_gain: List[numpy.ndarray] = [
        numpy.where((data & 2 ** 14 == 0) & (data & 2 ** 15 == 0)),
        numpy.where((data & (2 ** 14) > 0) & (data & 2 ** 15 == 0)),
        numpy.where(data & 2 ** 15 > 0),
    ]
    
    gain_mode: int
    
    for gain_mode in range(3):
        
        corrected_data[where_gain[gain_mode]] -= dark[gain_mode][where_gain[gain_mode]]
        
        corrected_data[where_gain[gain_mode]] /= (
            gain[gain_mode][where_gain[gain_mode]]
        )
        corrected_data[numpy.where(dark[0] == 0)] = 0

    return corrected_data

def main():
    global dark, gain

    # argument parser
    parser = argparse.ArgumentParser(
        description="Decode and sum up Jungfrau images.")
    parser.add_argument("-p1", "--pedestal1", type=str, action="store",
        help="path to the pedestal file for module 1")
    parser.add_argument("-p2", "--pedestal2", type=str, action="store",
        help="path to the pedestal file for module 2")
    parser.add_argument("-g1", "--gain1", type=str, action="store",
        help="path to the gain info file for module 1")
    parser.add_argument("-g2", "--gain2", type=str, action="store",
        help="path to the gain info file for module 1")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-m", "--mask", type=str, action="store",
        help="path to the virtual H5 mask file")
    parser.add_argument("-b", "--begin_frame", type=int, action="store",
        help="process frames starting from b ")
    parser.add_argument("-n", "--n_frames", type=int, action="store",
        help="process next n frames")
    parser.add_argument("-f", "--fast", type=int, action="store",
        help="fast center finding process")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args()

    #check arguments
    if not (args.pedestal1 and args.pedestal2
            and args.gain1 and args.gain2
            and args.input and args.output):
        print("missing argument")
        return

    #import dark and gain files
    num_panels: int = 2
    dark_filenames = [args.pedestal1, args.pedestal2]
    gain_filenames = [args.gain1, args.gain2]
    dark = numpy.ndarray(
        (3, 512 * num_panels, 1024), dtype=numpy.float32)
    gain = numpy.ndarray(
        (3, 512 * num_panels, 1024), dtype=numpy.float64)
    panel_id: int
    for panel_id in range(num_panels):
        gain_file: BinaryIO = open(gain_filenames[panel_id], "rb")
        dark_file: Any = h5py.File(dark_filenames[panel_id], "r")
        gain_mode: int
        for gain_mode in range(3):
            dark[gain_mode, 512 * panel_id : 512 * (panel_id + 1), :] = dark_file[
                "gain%d" % gain_mode][:]
            gain[gain_mode, 512 * panel_id : 512 * (panel_id + 1), :] = numpy.fromfile(
                    gain_file,
                    dtype=numpy.float64,
                    count=1024 * 512
                ).reshape(512, 1024)
        gain_file.close()
        dark_file.close()

    #loop through series
    with h5py.File(args.input, "r") as f:
        #n_frames = f["data"].shape[0]
        n_frames =args.n_frames
        begin= args.begin_frame
        x = f["entry/data/data"].shape[1]
        y = f["entry/data/data"].shape[2]

        print('%s frames %sx%s' % (n_frames, x, y))
        phase = 0
        i = 0
        acc_image = numpy.zeros((n_frames, y, x), dtype=numpy.float64)
        print('Applying calibration')

        with Pool(10) as p:
            d=p.map(apply_calibration, f["entry/data/data"][begin:begin+n_frames])

        for idx, i in enumerate(d):
            acc_image[idx] = i.reshape(y, x)

    ctr_img_lst=[]
    data=acc_image[:n_frames]
    
    if args.fast==1:

        print('Center finding search com')
        ###correct center
        f= h5py.File(args.mask, "r")
        mask=center_finding.include_gaps_np(numpy.array(f['data/data']))
        f.close()
        center=[]

        for idx, i in enumerate(data):
            print(f'Frame: {idx}')
            center.append(center_finding.calc_com_iter_mask(i,mask))

    else:
        print('Center finding search cc')
        with Pool(10) as p:
            center=p.map(center_finding.calc_center_cc_par, data)

    print('Applying geometry correction')
    acc_corr_image=[]

    for i in data:
        acc_corr_image.append(center_finding.include_gaps_np(i))

    print('Applying center finding correction')
    data=acc_corr_image[:n_frames]
    for idx,i in enumerate(data):
        new_data, shift=center_finding.bring_center_to_point(i, center[idx])
        ctr_img_lst.append(new_data)

    center_x=[]
    center_y=[]
    for i in center:
        center_x.append(i[0])
        center_y.append(i[1])

    print(f'Sum up {n_frames} frames')
    sum_img=sum_nth(numpy.array(ctr_img_lst),1,0)
    print(f'Azimuthal integration')
    rad_signal=radial_average(sum_img, None)


    print('Applying center finding correction masking data')

    f= h5py.File(args.mask, "r")
    mask=center_finding.include_gaps_np(numpy.array(f['data/data']))
    f.close()
    data=acc_corr_image[:n_frames]
    mask_ctr_img_lst=[]

    for idx,i in enumerate(data):
        masked_data=mask.copy()
        new_data, shift=center_finding.bring_center_to_point(masked_data, center[idx])
        mask_ctr_img_lst.append(new_data)


    print(f'Sum up {n_frames} frames mask')
    sum_img_masked=sum_nth_mask(numpy.array(mask_ctr_img_lst),1,0)
    print(f'Azimuthal integration')
    #rad_signal_masked=radial_average(sum_img, sum_img_masked)
    masked_data_final=sum_img_masked*sum_img
    #plt.imshow(masked_data_final,vmax=1e4)
    #plt.show()
    rad_signal_masked=azimuthal_average(masked_data_final, center=(515,532),angular_bounds=(0,360),trim=True)
    rad_signal_cut=[]
    beam_cut=0
    rad_signal_cut.append(numpy.array(rad_signal_masked[0][beam_cut:]))
    rad_signal_cut.append(numpy.array(rad_signal_masked[1][beam_cut:]))

    baseline=utils.baseline_als(rad_signal_cut[1],1e4, 0.1)
    rad_sub=rad_signal_cut[1]-baseline
    rad_sub[numpy.where(rad_sub<0)]=0
    #plt.plot(rad_signal_cut[1])
    #plt.plot(baseline)
    #plt.plot(rad_sub)
    #plt.show()
    peaks, half=utils.calc_fwhm(rad_sub,6,threshold=100,distance=5, height=5, width=3)
    peak_px=peaks+rad_signal_cut[0][0]
    fwhm_over_rad=[]
    intensity=[]
    for i in range(len(peaks)):
        fwhm_over_rad.append(half[0][i]/peak_px[i])
        intensity.append(rad_sub[peaks[i]])
    print(peaks,fwhm_over_rad, half[0], intensity)
    #plt.plot(rad_sub)
    #plt.scatter(peaks,intensity,c='r')
    #plt.show()

    angles, peak_positions=utils.ellipse_measure(masked_data_final, beam_cut)

    #write output
    f= h5py.File(args.output+'.h5', "w")
    f.create_dataset('data',data=acc_corr_image)
    f.create_dataset('centered_data',data=ctr_img_lst)
    f.create_dataset('sum_frames',data=sum_img)
    f.create_dataset('rad_sub',data=rad_sub)
    f.create_dataset('sum_frames_mask',data=masked_data_final)
    f.create_dataset('rad_average_mask',data=rad_signal_cut[1])
    f.create_dataset('rad_x',data=rad_signal_cut[0])
    f.create_dataset('intensity',data=intensity)
    f.create_dataset('fwhm_radius',data=fwhm_over_rad)
    f.create_dataset('fwhm',data=half[0])
    f.create_dataset('peak_position',data=peak_px)

    f.create_dataset('ellipse_angle',data=angles)
    f.create_dataset('ellipse_peak',data=peak_positions)
    f.close()


if __name__ == '__main__':
    main()
