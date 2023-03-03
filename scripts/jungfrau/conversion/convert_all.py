"""
Function apply_calibration is based on OnDA - Deutsches Elektronen-Synchrotron DESY,
a research centre of the Helmholtz Association.
"""

import h5py
import argparse
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as color

dark = None
gain = None

def filter_data(data):
    """
    Filter JUNGFRAU 1M faulty images based on number of pixels at gain 2.

    Parameters
    ----------
    data: np.ndarray
        JUNGFRAU 1M single raw image

    Returns
    ----------
    bool
        True if file should be skipped before apply calibration
    """
    gain_3=np.where(data & 2 ** 15 > 0)
    counts=gain_3[0].shape[0]
    if counts>1e4:
        return 1
    else:
        return 0


def apply_calibration(data: np.ndarray) -> np.ndarray:
    """
    Applies the calibration to a JUNGFRAU 1M detector data frame.

    This function determines the gain stage of each pixel in the provided data
    frame, and applies the relevant gain and offset corrections.

    Parameters:

        data: The detector data frame to calibrate.

    Returns:

        The corrected data frame.
    """
    corrected_data: np.ndarray = data.astype(np.float32)
    
    where_gain: List[np.ndarray] = [
        np.where((data & 2 ** 14 == 0) & (data & 2 ** 15 == 0)),
        np.where((data & (2 ** 14) > 0) & (data & 2 ** 15 == 0)),
        np.where(data & 2 ** 15 > 0),
    ]
    
    gain_mode: int
    
    for gain_mode in range(3):
        
        corrected_data[where_gain[gain_mode]] -= dark[gain_mode][where_gain[gain_mode]]
        
        corrected_data[where_gain[gain_mode]] /= (
            gain[gain_mode][where_gain[gain_mode]]
        )
        corrected_data[np.where(dark[0] == 0)] = 0

    return corrected_data


def main(raw_args=None):
    parser = argparse.ArgumentParser(
    description="Convert JUNGFRAU 1M H5 images collected at REGAE for rotational data step/fly scan and return images in rotation sequence according tro the file index.")
    parser.add_argument("-i", "--input", type=str, action="store",
    help="hdf5 input image")
    parser.add_argument("-s", "--start_index", type=int, action="store",
    help="starting file index")
    parser.add_argument("-e", "--end_index", type=int, action="store",
    help="ending file index")
    parser.add_argument("-f", "--frames", default=None, type=int, action="store",
    help="If more than one frame was measured per step. Number of frames to be accumulated per step for rotational step manner. None for fly scan.")
    parser.add_argument("-p1", "--pedestal1", type=str, action="store",
        help="path to the pedestal file for module 1")
    parser.add_argument("-p2", "--pedestal2", type=str, action="store",
        help="path to the pedestal file for module 2")
    parser.add_argument("-g1", "--gain1", type=str, action="store",
        help="path to the gain info file for module 1")
    parser.add_argument("-g2", "--gain2", type=str, action="store",
        help="path to the gain info file for module 2")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="hdf5 output path")
    args = parser.parse_args(raw_args)

    global dark, gain

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

    
    index=np.arange(args.start_index,args.end_index,1)
    n_frames=args.end_index-args.start_index
    
    d=np.zeros((n_frames, 1024, 1024),dtype=np.int32)

    for idx,i in enumerate(index):
        #print(i)
        acc_frame=np.zeros((1024, 1024),dtype=np.int32)
        
        f=h5py.File(f"{args.input}_master_{i}.h5", "r")

        size=len(f["entry/data/data"])
        try:
            raw=np.array(f["entry/data/data"][:size])
        except OSError:
            print('skipped',i)
            continue

        if args.frames==None:
            n_frames_measured=raw.shape[0]
        else:
            n_frames_measured=args.frames
        corr_frame=np.zeros((n_frames_measured,1024, 1024), dtype=np.int32)
        
        f.close()
        for idy,j in enumerate(raw[:n_frames_measured]):
            skip=filter_data(j)
            if skip==0:
                corr_frame[idy]=apply_calibration(j)
                acc_frame+=corr_frame[idy]            

        d[idx]=acc_frame
    #print(d.shape)
    g=h5py.File(args.output+'.h5', "w")
    g.create_dataset('data',data=d)
    g.close()
     
if __name__ == '__main__':
    main()
