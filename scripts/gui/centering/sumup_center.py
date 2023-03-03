# -*- coding: utf-8 -*-

import crystfel_geometry
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
from skued import gaussian
from skued import baseline_dwt,baseline_dt
import peakutils
dark = None
gain = None


def filter_data(data):
    gain_3=numpy.where(data & 2 ** 15 > 0)
    counts=gain_3[0].shape[0]
    if counts>1e4:
        return 1
    else:
        return 0


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


def main(raw_args=None):
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
    parser.add_argument("-g", "--geom", type=str, action="store",
        help="path to the virtual H5 mask file")


    args = parser.parse_args(raw_args)

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

        #print('%s frames %sx%s' % (n_frames, x, y))
        raw_data=f["entry/data/data"][begin:begin+n_frames]
        filt_data=raw_data.copy()
        for i in range(n_frames):
             skip=filter_data(filt_data[i])
             if skip==1:
                 filt_data[i]=dark[0]
        acc_image = numpy.zeros((n_frames, y, x), dtype=numpy.float64)
        #print('Applying calibration')

        d=numpy.zeros((n_frames, y, x), dtype=numpy.float64)
        for idx, i in enumerate(filt_data):
             d[idx]=apply_calibration(i)

        for idx, i in enumerate(d):
            acc_image[idx] = i.reshape(y, x)

    
    data=acc_image[:n_frames]

    geometry_filename=args.geom
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2
        * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min())))
        + 2
    )
    x_minimum: int = (
        2
        * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min())))
        + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)


    f= h5py.File(args.mask, "r")
    mask=numpy.array(f['data/data'])
    f.close()

    masked_data=[crystfel_geometry.apply_geometry_to_data(i*mask,geometry) for i in data]
    center_p0=[_img_center_x,_img_center_y]

    ctr_img_lst=[]
    sum_img_data=sum_nth(numpy.array(masked_data),1,0)

    if args.fast==0:
        print('No centering between frames')
        ###calc_com of sum image create center list with all centers the same of com


    if args.fast==1:

        print('Center finding search com')
        ###correct center
        
        center=[]

        for idx, i in enumerate(masked_data):
            print(f'Frame: {idx}')
            center.append(center_finding.calc_com(i))

    elif args.fast==2:
        print('Center finding search difference of com flip image')
        
        with Pool(20) as p:
            center=p.map(center_finding.calc_center_com_par, masked_data)

    elif args.fast==3:
        print('Center finding search cc summed image')
        for idx, i in enumerate(masked_data):
            print(f'Frame: {idx}')
            center.append(center_finding.calc_center_cc_sum(i,sum_img_data))

    elif args.fast==4:
        print('Center finding search cc flip image')
        
        with Pool(20) as p:
            center=p.map(center_finding.calc_center_cc_par, masked_data)

    elif args.fast==5:
        print('Center finding shift and calc cc flip image')
        with Pool(20) as p:
            center=p.map(center_finding.calc_center_corr_coef_par, masked_data)

    elif args.fast==6:
        print('Center finding shift and calc cc sum image')
        for idx, i in enumerate(masked_data):
            print(f'Frame: {idx}')
            center.append(center_finding.calc_center_corr_coef_sum_par(i,sum_img_data))


    for idx,i in enumerate(masked_data):
        new_data, shift=center_finding.bring_center_to_point(i, center[idx])
        ctr_img_lst.append(new_data)

    print(f'Sum up {n_frames} frames')
    sum_img=sum_nth(numpy.array(ctr_img_lst),1,0)
   

    f= h5py.File(args.output+'.h5', "a")
    f.create_dataset('data', data=data)
    f.create_dataset('center_data', data=center)
    f.create_dataset('centered_data', data=ctr_img_lst)
    f.create_dataset('sum_centered', data=sum_img)
    f.create_dataset('sum', data=sum_img_data)
    f.close()

if __name__ == '__main__':
    main()
