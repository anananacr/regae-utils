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
    parser = argparse.ArgumentParser(
        description="Find peak positions according to peak range.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-p", "--peak_range", action='append',
        help="path to the virtual H5 mask file")
    args = parser.parse_args(raw_args)
    
    peak_range=[]
    list_of_peaks=[]
    for idx,i in enumerate(args.peak_range):
        peak_range.append(int(i))
        if idx%2!=0:
            peak_range.sort()
            list_of_peaks.append(peak_range)
            peak_range=[]
    f= h5py.File(args.input, "r+")
    rad_sub=[numpy.array(f['radial_x']),numpy.array(f['radial'])]
    rad_sub=numpy.transpose(numpy.array(rad_sub))
    #print(list_of_peaks)
    peak_px=[]
    intensities=[]

    for i in list_of_peaks:
        index=[numpy.where(rad_sub[:,0]==i[0])[0][0],numpy.where(rad_sub[:,0]==i[1])[0][0]]
        #print(index)
        rad_signal_cut=[rad_sub[:,0][index[0]:index[1]],rad_sub[:,1][index[0]:index[1]]]
        #plt.plot(rad_sub[:,0], rad_sub[:,1])
        #plt.plot(rad_signal_cut[0],rad_signal_cut[1],'r:')
        #plt.show()

        peak_pos, half=utils.calc_fwhm(rad_signal_cut[1],-1,threshold=100,distance=4, height=0, width=2)
        print(peak_pos)
        while len(peak_pos)==0:
            index[0]-=1
            index[1]+=1
            rad_signal_cut=[rad_sub[:,0][index[0]:index[1]],rad_sub[:,1][index[0]:index[1]]]
            peak_pos, half=utils.calc_fwhm(rad_signal_cut[1],-1,threshold=1,distance=4, height=0, width=2)

        if len(peak_pos)>1:
            maxim=0
            for k in peak_pos:
                 if (rad_signal_cut[1][k])>maxim:
                     max_peak=k
                     maxim=rad_signal_cut[1][k]
            peak_pos=[max_peak]

        peak_px.append(peak_pos+rad_signal_cut[0][0])
        
    print(peak_px)
    #plt.plot(rad_sub[:,0], rad_sub[:,1])
    #plt.scatter(peak_px,intensities,c='r')
    #plt.show()
    #print(peak_px)
    try:
        f.create_dataset('peak_position',data=peak_px)
        f.create_dataset('peak_range',data=numpy.array(list_of_peaks))
    except:
        del f['peak_position']
        del f['peak_range']
        f.create_dataset('peak_position',data=peak_px)
        f.create_dataset('peak_range',data=numpy.array(list_of_peaks))

    f.close()

if __name__ == '__main__':
    main()
