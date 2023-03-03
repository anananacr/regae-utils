import h5py
import numpy as np
import argparse
from utils import sum_nth_mask,radial_average
from powder import azimuthal_average

def main():
    parser=argparse.ArgumentParser(
        description="Merge summed up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to H5 data files")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args()
    
    #radial_averages=np.zeros((600,))

    f=h5py.File(args.input, "r")
    data=np.array(f['/data'])
    #rad_x,radial_averages=radial_average(data,None)
    rad_signal=azimuthal_average(data, center=(515,532),angular_bounds=(50,150),trim=True)
    rad_x=rad_signal[0]
    radial_averages=rad_signal[1]
    f.close()

    f= h5py.File(args.output, "w")
    f.create_dataset('radial',data=radial_averages)
    f.create_dataset('radial_x',data=rad_x)
    f.close()

if __name__ == '__main__':
    main()
