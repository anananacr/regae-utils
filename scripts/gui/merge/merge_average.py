import h5py
import numpy as np
import argparse


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Take the mean of every element in the input file and write in output."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="path to H5 data files"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output H5 file"
    )

    args = parser.parse_args(raw_args)
    mean = []

    f = h5py.File(args.input + ".h5", "r")
    keys = list(f.keys())
    print(keys)

    g = h5py.File(args.output + ".h5", "w")

    for i in keys:
        data = np.array(f[i])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        g.create_dataset(i, data=mean)
        g.create_dataset(i + "_err", data=std)

    f.close()
    g.close()


if __name__ == "__main__":
    main()
