import h5py
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot time scans from scan_radial_to_file.py states on and off."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output label for. png image"
    )
    args = parser.parse_args()

    ## Set time scans points to be plotted together
    size = [-46.7, -20.0, -6.7, -0.0, 6.7, 20.0, 46.7]
    size.sort()
    fh, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_facecolor("grey")

    n = round(len(size))
    colors = pl.cm.jet(np.linspace(0.1, 0.9, n))
    for idx, i in enumerate(size):
        file_path = f"{args.input}"
        hf = h5py.File(file_path, "r")
        data_name = f"time_scan_{i}_laser_off"
        data = np.array(hf[data_name])
        norm_rad_0 = data[:, 1]
        x = data[:, 0]
        plt.plot(x, norm_rad_0, ":", color=colors[idx], label=f"{i}ps laser off")
        hf.close()
        file_path = f"{args.input}"
        hf = h5py.File(file_path, "r")
        data_name = f"time_scan_{i}_laser_on"
        data = np.array(hf[data_name])
        norm_rad_0 = data[:, 1]
        x = data[:, 0]
        plt.plot(x, norm_rad_0, ".-", label=f"{i}ps laser on", color=colors[idx])
        hf.close()

    ax.set_xlabel("Radius [pixel]")
    ax.set_ylabel("Intensity [keV]")
    ax.grid()
    ax.set_xlim(30, 120)
    plt.legend()
    plt.savefig(args.output + ".png")
    plt.show()


if __name__ == "__main__":
    main()
