import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot peak positions according to angle for first n peaks set on n."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-n", "--n_peaks", type=int, action="store", help="number of peaks to plot"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output label for. png image"
    )
    args = parser.parse_args()

    fh, ax = plt.subplots(1, 1, figsize=(8, 8))

    size = np.arange(0, args.n_peaks + 1, 1)
    label = ["peak_0", "peak_1", "peak_2", "peak_3"]
    count = 0
    colors = ["b", "b", "b", "b"]
    file_path = args.input
    hf = h5py.File(file_path, "r")
    data_name = f"ellipse_peak"
    rad_0 = np.array(hf[data_name], dtype=float)
    x = np.array(hf[f"ellipse_angle"])
    hf.close()

    for i in size:
        for j in range(len(colors)):
            plt.scatter(
                x[:], rad_0[:, j], label=f"{label[j]}", color=colors[j], marker="."
            )
            data = rad_0[:, j].copy()
            data[np.where(data == 0)] = np.nan
            mean = (np.nanmean(data)) * np.ones(350)
            textstr = f"Mean:{round(np.nanmean(data),2)}\nMedian:{round(np.nanmedian(data),2)}\nStd:{round(np.nanstd(data),2)}"
            # ax.text(0.05, 0.4*j+0.15, textstr, transform=ax.transAxes, fontsize=8,verticalalignment='top')
            print(textstr)
            # print(mean)
            # plt.plot(mean,color=colors[j])

    ax.set_xlabel("$\Theta$ [pixel]")
    ax.set_ylabel("Peak distance from center [pixels]")
    ax.grid(which="both")
    # textstr=f'Mean:{round(np.mean(rad_0),2)}\nMedian:{round(np.median(rad_0),2)}\nStd:{round(np.std(rad_0),2)}\nVar:{round(np.var(rad_0),2)}'
    # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,verticalalignment='top')
    # print(textstr)
    # ax.set_xlim(0,350)
    ax.set_ylim(50, 200)
    # plt.legend()
    plt.savefig(args.output + ".png")
    plt.show()


if __name__ == "__main__":
    main()
