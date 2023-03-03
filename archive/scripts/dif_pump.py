import h5py
import numpy as np
import argparse
from utils import sum_nth_mask


def main():
    parser=argparse.ArgumentParser(
        description="Merge summed up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to H5 data files")
    parser.add_argument("-z", "--zero", type=str, action="store",
        help="path to H5 data files ")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args()
    diff=np.zeros((1064,1030))

    f=h5py.File(args.input, "r")
    data=np.array(f['/sum_frames_mask'])
    g=h5py.File(args.zero, "r")
    zero=-1*np.array(g['/sum_frames_mask'])

    diff=sum_nth_mask(np.array([data,zero]),1,0)
    f.close()
    g.close()

    f= h5py.File(args.output+'.h5', "w")
    f.create_dataset('data',data=diff)
    f.close()

if __name__ == '__main__':
    main()
