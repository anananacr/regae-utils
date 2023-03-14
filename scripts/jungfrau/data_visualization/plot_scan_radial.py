import h5py
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot radial average for different Time delays in pump and probe experiments, states on and off."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output label for. png image"
    )
    args = parser.parse_args()

    ## number of file index
    size = np.arange(6, 24, 1)
    # size=[78,79,40,41,36,37,32,33,30,31,24,25,2,3]

    # pl.style.use('seaborn')
    fh, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_facecolor("grey")

    ## corresponding labels
    label = ["-80ps", "-16.65ps", "-9.99ps", "-3.33ps", "0", "9.9ps", "46.7ps"]
    # label=['3','5','7','9']

    n = round(len(size) / 2)
    colors = pl.cm.jet(np.linspace(0.1, 0.9, n))
    for idx, i in enumerate(size):
        file_path = f"{args.input}_{i}.h5"
        hf = h5py.File(file_path, "r")
        ## hdf5 path to radial average intensities and radius in px (rad_x)
        data_name = "rad_average_mask"
        rad_0 = np.array(hf[data_name])
        norm_rad_0 = rad_0 / np.max(rad_0)
        x = np.array(hf["rad_x"])
        hf.close()
        # plt.plot(x,rad_0, label =f'{label[idx]}')
        if idx % 2 == 0:
            plt.plot(
                x,
                norm_rad_0,
                ":",
                color=colors[round(idx / 2)],
                label=f"{label[round(idx/2)]} laser off",
            )
        else:
            print("skip")
            plt.plot(
                x,
                norm_rad_0,
                ".-",
                label=f"{label[round((idx-1)/2)]} laser on",
                color=colors[round((idx - 1) / 2)],
            )

    ax.set_xlabel("Radius [pixel]")
    ax.set_ylabel("Intensity [keV]")
    ax.grid()
    ax.set_xlim(30, 180)
    plt.legend()
    # plt.savefig(args.output+'.png')
    plt.show()


if __name__ == "__main__":
    main()
