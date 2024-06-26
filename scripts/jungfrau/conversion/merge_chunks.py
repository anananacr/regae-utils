import h5py
import numpy as np
import argparse
import glob


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate converted images in packets form convert_images.py, when unequal packets were created. Set size of packets hard coded manually."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="path to H5 data files"
    )
    parser.add_argument(
        "-s", "--start", type=int, action="store", help="start image number"
    )
    parser.add_argument(
        "-e", "--end", type=int, action="store", help="final image numebr"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output H5 file"
    )

    args = parser.parse_args()
    #index = np.arange(args.start, args.end, args.packet)
    index = [16999]

    #total = args.end - args.start
    total = sum(index)

    convert_images = np.zeros((total, 1024, 1024), dtype=np.int32)
    count = 0

    for idx, i in enumerate(index):
        f = h5py.File(f"{args.input}_{idx}.h5", "r")
        data = np.array(f["/entry/data/data"])
        convert_images[count : count + i] = data
        f.close()
        count += i

    f = h5py.File(args.output + ".h5", "w")
    f.create_dataset("/entry/data/data", data=convert_images, compression="gzip")
    f.close()


if __name__ == "__main__":
    main()
