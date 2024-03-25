import h5py
import numpy as np
import argparse


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Take the mean of every element in the input file and write in output. Concatenates radial average."
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

    g = h5py.File(args.output + ".h5", "w")

    for i in keys[:-3]:
        data = np.array(f[i])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        g.create_dataset(i, data=mean)
        g.create_dataset(i + "_err", data=std)

    ## Allignement of radial averages
    data = np.array(f["intensity"])
    n_files = data.shape[0]
    name = ["rad_average_mask"]
    for k in name:
        rad_ave = np.zeros((n_files, 550))
        rad = np.array(f[k])
        rad_x = np.array(f["rad_x"], dtype=int)
        scan_number = 0

        for i in range(n_files):
            # print(i, rad[:][i])
            radial_plot = rad[i][:]

            x_radial_plot = rad_x[i][:]
            for idy, j in enumerate(x_radial_plot[:-1]):
                rad_ave[i][j] = radial_plot[idy]

        rad_ave[rad_ave == 0] = np.nan
        means = np.nanmean(rad_ave[:500], axis=0)
        means = np.nan_to_num(means)
        std = np.nanstd(rad_ave[:500], axis=0)
        std = np.nan_to_num(std)
        g.create_dataset(k, data=means)
        g.create_dataset(k + "_err", data=std)
    g.create_dataset("rad_x", data=np.arange(0, 500))

    f.close()
    g.close()


if __name__ == "__main__":
    main()
