import h5py
import numpy as np
import argparse
import sys
sys.path.append("../../utils/")
from utils import sum_nth_mask

def main():
    parser=argparse.ArgumentParser(
        description="Merge summed up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to H5 data files")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args()
    ctr_images=[]
    radial_averages=[]
    for i in range(0,5):
        f=h5py.File(args.input+str(100*(i))+'.h5', "r")
        data=np.array(f['/sum_frames_mask'])
        ctr_images.append(data)
        radial_averages.append(np.array(f['/rad_average_mask']))
        f.close()
    #sum_ctr=sum_nth_mask(np.array(ctr_images),1,0)

    f= h5py.File(args.output+'.h5', "w")
    f.create_dataset('data',data=ctr_images)
    f.create_dataset('radial',data=radial_averages)
    f.close()

if __name__ == '__main__':
    main()
