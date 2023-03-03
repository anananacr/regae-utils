import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt

def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot pixel intensity histogram over files measured for single electron detection.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-px", "--pixel_x", type=int, action="store",
        help="x/collum pixel position")
    parser.add_argument("-py", "--pixel_y", type=int, action="store",
        help="y/row pixel position")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")

    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:
        data=np.array(f["raw_data"])

    fig, ax = plt.subplots(1,1, figsize=(8, 8))
    values=np.array(data.shape[0])
    thr=10
    values=[i[args.pixel_y,args.pixel_x] for i in data if i[args.pixel_y,args.pixel_x]>thr]
    _ = plt.hist(values, bins=120,range=(0,300))
    ax.set_xlabel('Edep/e- [keV]')
    ax.set_ylabel('Counts')
    ax.set_title(f'x = {args.pixel_x} y= {args.pixel_y}')
    #plt.show()
    plt.savefig(f'{args.output}.png')

        
if __name__ == '__main__':
    main()
