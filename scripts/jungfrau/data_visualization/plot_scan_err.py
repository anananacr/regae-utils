import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot pump and probe scan intensity and FWHM, states on/off, but no error bars."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output label for. png image"
    )
    args = parser.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ## number of file index
    size = np.flip(np.arange(1, 83, 2))

    ## final motor position in mm and t0 motor position in mm
    start = -1 * (240 - 227.5) * 6.66

    ## t0 motor position in mm and start motor position in mm
    finish = (227.5 - 220) * 6.66 + 0.5
    label = np.arange(start, finish, 3.33)

    # size=np.arange(0,83,1)
    # label=np.arange(0,83,1)
    # print(label)

    data_name = "intensity"
    data_name_0 = "fwhm_radius"
    color = ["orange", "r", "g", "g", "b", "g"]
    count = 0

    ### Set peaks to plot
    peaks_to_plot = range(1, 4)

    for idx, i in enumerate(size):
        print(i)
        x = [label[idx], label[idx], label[idx], label[idx], label[idx], label[idx]]

        file_path = f"{args.input}_{i}.h5"
        hf = h5py.File(file_path, "r")
        rad_0 = np.array(hf[data_name])
        rad_0_err = np.array(hf[data_name + "_err"])
        rad_1 = np.array(hf[data_name_0])
        rad_1_err = np.array(hf[data_name_0 + "_err"])

        if count == 0:
            print("hey")
            # norm_rad_0=rad_0.copy()
            # norm_rad_1=rad_1.copy()
            norm_rad_0 = [1, 1, 1, 1, 1]
            norm_rad_1 = [1, 1, 1, 1, 1]
        if i % 2 == 0:
            ax1 = plt.subplot(121)
            for j in peaks_to_plot:
                # print(rad_0[j], norm_rad_0)
                norm_int = round((rad_0[j] / norm_rad_0[j]), 6)
                norm_int_err = round((rad_0_err[j] / norm_rad_0[j]), 6)
                if count == 0:
                    ax1.scatter(
                        x[j],
                        norm_int,
                        color=color[j],
                        label=f"peak_{j}_off",
                        marker="o",
                    )
                    ax1.errorbar(x[j], norm_int, yerr=norm_int_err, color=color[j])
                else:
                    ax1.scatter(x[j], norm_int, color=color[j], marker="o")
                    ax1.errorbar(x[j], norm_int, yerr=norm_int_err, color=color[j])
            ax2 = plt.subplot(122)
            for j in peaks_to_plot:
                # print(rad_1[j], norm_rad_1)
                norm_fwhm = round((rad_1[j] / norm_rad_1[j]), 6)
                norm_fwhm_err = round((rad_1_err[j] / norm_rad_1[j]), 6)
                if count == 0:
                    ax2.scatter(
                        x[j],
                        norm_fwhm,
                        color=color[j],
                        label=f"peak_{j}_off",
                        marker="o",
                    )
                    ax2.errorbar(x[j], norm_fwhm, yerr=norm_fwhm_err, color=color[j])
                else:
                    ax2.scatter(x[j], norm_fwhm, color=color[j], marker="o")
                    ax2.errorbar(x[j], norm_fwhm, yerr=norm_fwhm_err, color=color[j])
        else:
            ax1 = plt.subplot(121)
            for j in peaks_to_plot:
                # print(rad_0[j], norm_rad_0)
                norm_int = round((rad_0[j] / norm_rad_0[j]), 6)
                norm_int_err = round((rad_0_err[j] / norm_rad_0[j]), 6)
                if count == 1:
                    ax1.scatter(
                        x[j],
                        norm_int,
                        color=color[j + 1],
                        label=f"peak_{j}_on",
                        marker="o",
                    )
                    ax1.errorbar(x[j], norm_int, yerr=norm_int_err, color=color[j + 1])
                else:
                    ax1.scatter(x[j], norm_int, color=color[j + 1], marker="o")
                    ax1.errorbar(x[j], norm_int, yerr=norm_int_err, color=color[j + 1])
            ax2 = plt.subplot(122)
            for j in peaks_to_plot:
                # print(rad_1[j], norm_rad_1)
                norm_fwhm = round((rad_1[j] / norm_rad_1[j]), 6)
                norm_fwhm_err = round((rad_1_err[j] / norm_rad_1[j]), 6)
                if count == 0:
                    ax2.scatter(
                        x[j],
                        norm_fwhm,
                        color=color[j + 1],
                        label=f"peak_{j}_on",
                        marker="o",
                    )
                    ax2.errorbar(
                        x[j], norm_fwhm, yerr=norm_fwhm_err, color=color[j + 1]
                    )
                else:
                    ax2.scatter(x[j], norm_fwhm, color=color[j + 1], marker="o")
                    ax2.errorbar(
                        x[j], norm_fwhm, yerr=norm_fwhm_err, color=color[j + 1]
                    )

        hf.close()
        count += 1

    ax1.legend(labels=["peak_1:64_laser_off", "peak_1:64_laser_on"])
    # ax2.legend()
    ax1.set_xlabel("Time delay (ps)")
    # ax1.set_xlabel('Index file')
    ax1.set_ylabel("Normal. intensity")
    # ax1.set_ylim(0.8,1.2)
    # ax2.set_ylim(0.9,1.1)
    ax2.set_xlabel("Time delay stage (ps)")
    # ax2.set_xlabel('Index file')
    ax2.set_ylabel("FWHM/Radius")
    ax1.grid()
    ax2.grid()
    # plt.savefig(args.output+'.png')
    plt.show()


if __name__ == "__main__":
    main()
