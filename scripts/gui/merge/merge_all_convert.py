import h5py
import numpy
import argparse
from utils import sum_nth_mask
import glob

def main():
    parser=argparse.ArgumentParser(
        description="Merge summed up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to H5 data files")
    parser.add_argument("-s", "--start", type=int, action="store",
        help="path to H5 data files")
    parser.add_argument("-e", "--end", type=int, action="store",
        help="path to H5 data files")
    parser.add_argument("-p", "--packet", type=int, action="store",
        help="path to H5 data files")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args()
    index=numpy.arange(args.start,args.end,args.packet)
    total=args.end-args.start
    
    convert_images=numpy.zeros((total,1024, 1024), dtype=numpy.float64)
    count=0

    for idx,i in enumerate(index):
        f=h5py.File(f"{args.input}_{i}.h5", "r")
        data=numpy.array(f['/data'])
        convert_images[count:count+args.packet]=data
        f.close()
        count+=args.packet
    #sum_ctr=sum_nth_mask(numpy.array(ctr_images),1,0)

    f= h5py.File(args.output+'.h5', "w")
    f.create_dataset('data',data=convert_images)
    f.close()

if __name__ == '__main__':
    main()
