import h5py
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Sum up Jungfrau data from different H5 files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="files list with paths to H5 data files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        action="store",
        help="output path to H5 summed data file",
    )
    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()

    count = 0
    for i in paths:
        hdf5_file = str(i[:-1])
        f = h5py.File(hdf5_file, "r")
        # data=f['entry/data/data']
        data = f["data"]
        if count == 0:
            acc = np.zeros((data[0].shape))
        for j in data:
            acc += j

    g = h5py.File(args.output, "w")
    g.create_dataset("data", data=acc)
    g.close()


if __name__ == "__main__":
    main()
