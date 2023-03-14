import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot fitted R-squares in range of data."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to the H5 data master file",
    )

    args = parser.parse_args(raw_args)

    files = list(glob.glob(f"{args.input}*.h5"))

    fit_0 = []
    fit_1 = []

    for i in files:
        with h5py.File(i, "r") as f:
            data = np.array(f["fit_r_squared"])
            fit_0.append(data[0])
            fit_1.append(data[1])

    plt.plot(fit_0, "o-")
    plt.plot(fit_1, "o-")

    plt.show()


if __name__ == "__main__":
    main()
