import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit


def fit(x, a, b):
    return a * x + b


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot pump and probe scan intensity and FWHM, states on or off, substituting points by the mean running average."
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
    size = np.flip(np.arange(4, 85, 2))

    ## final motor position in mm and t0 motor position in mm
    start = -1 * (230 - 217) * 6.66

    ## t0 motor position in mm and start motor position in mm
    finish = (217 - 210) * 6.66 + 0.5
    label = np.arange(start, finish, 6.66)
    print(label)

    data_name = "intensity"
    data_name_0 = "fwhm_radius"
    color = ["r", "g", "b", "orange", "magenta", "purple"]
    count = 0

    int_peak_1 = []
    int_peak_2 = []
    fwhm_peak_1 = []
    fwhm_peak_2 = []

    for idx, i in enumerate(size):

        file_path = f"{args.input}_{i}.h5"
        hf = h5py.File(file_path, "r")
        rad_0 = np.array(hf[data_name])
        rad_1 = np.array(hf[data_name_0])
        hf.close()
        if count == 0:
            print("hey")
            norm_rad_0 = rad_0.copy()
            norm_rad_1 = rad_1.copy()
        ax1 = plt.subplot(121)
        int_peak_1.append(round((rad_0[1] / norm_rad_0[1]), 20))
        int_peak_2.append(round((rad_0[2] / norm_rad_0[2]), 20))
        fwhm_peak_1.append(round((rad_1[1] / norm_rad_1[1]), 20))
        fwhm_peak_2.append(round((rad_1[2] / norm_rad_1[2]), 20))

        hf.close()
        count += 1

    print(int_peak_1)

    mov_int_peak_1 = []
    mov_int_peak_2 = []
    mov_fwhm_peak_1 = []
    mov_fwhm_peak_2 = []

    ## Running average with 2 points
    for i in range(0, len(int_peak_1) - 1):
        if i % 2 == 0:

            mean = round(((int_peak_1[i] + int_peak_1[i + 1]) / 2), 6)
            mov_int_peak_1.append(mean)
            mean = round(((int_peak_2[i] + int_peak_2[i + 1]) / 2), 6)
            mov_int_peak_2.append(mean)
            mean = round(((fwhm_peak_1[i] + fwhm_peak_1[i + 1]) / 2), 6)
            mov_fwhm_peak_1.append(mean)
            mean = round(((fwhm_peak_2[i] + fwhm_peak_2[i + 1]) / 2), 6)
            mov_fwhm_peak_2.append(mean)

    ## Running average with 3 points
    """
    for i in range(1,len(int_peak_1)-1):
        if i%3==0:

            mean=round(((int_peak_1[i-1]+int_peak_1[i]+int_peak_1[i+1])/3),6)
            mov_int_peak_1.append(mean)
            mean=round(((int_peak_2[i-1]+int_peak_2[i]+int_peak_2[i+1])/3),6)
            mov_int_peak_2.append(mean)
            mean=round(((fwhm_peak_1[i-1]+fwhm_peak_1[i]+fwhm_peak_1[i+1])/3),6)
            mov_fwhm_peak_1.append(mean)
            mean=round(((fwhm_peak_2[i-1]+fwhm_peak_2[i]+fwhm_peak_2[i+1])/3),6)
            mov_fwhm_peak_2.append(mean)
    print(len(label),len(mov_int_peak_1),len(mov_int_peak_2), len(mov_fwhm_peak_1), len(mov_fwhm_peak_2))
    """

    ## Linear fitting
    popt, pcov = curve_fit(fit, label[1:], mov_int_peak_1, p0=(0.0, 0.0))
    a = popt[0]
    err_a = np.sqrt(pcov[0, 0])
    b = popt[1]
    err_b = np.sqrt(pcov[1, 1])
    perr = np.sqrt(np.diag(pcov))
    print(f"a={a}+/-{err_a},{abs(err_a/a*100.0)}%")
    print(f"b={b}+/-{err_b},{abs(err_b/b*100.0)}%")
    m_1, b_1 = np.polyfit(label[1:], mov_int_peak_1, 1)
    m_2, b_2 = np.polyfit(label[1:], mov_int_peak_2, 1)
    print(m_1, b_1)
    print(m_2, b_2)

    ### Set peaks range to be plotted
    peaks_to_plot = range(1, 2)
    for idx, i in enumerate(label[1:]):
        x = [i, i, i]

        ax1 = plt.subplot(121)
        for j in peaks_to_plot:
            if j == 1:
                ax1.scatter(x[j], mov_int_peak_1[idx], color=color[j], marker="o")
            else:
                ax1.scatter(x[j], mov_int_peak_2[idx], color=color[j], marker="o")

        x_fit = np.linspace(-90, 50, 1000)

        y_fit_1 = m_1 * x_fit + b_1
        # y_fit_2=m_2*x_fit+b_2
        plt.plot(x_fit, y_fit_1, "--k")
        # plt.plot(x_fit,y_fit_2,'--k')
        ax2 = plt.subplot(122)
        for j in peaks_to_plot:
            if j == 1:
                ax2.scatter(x[j], mov_fwhm_peak_1[idx], color=color[j], marker="o")
            else:
                ax2.scatter(x[j], mov_fwhm_peak_2[idx], color=color[j], marker="o")
        count += 1

    ax1.legend(labels=["peak_1", "peak_2"])
    # ax2.legend()
    ax1.set_xlabel("Time delay (ps)")
    ax1.set_ylabel("Intensity [keV]")
    # ax1.set_ylim(0.5,1.5)
    # ax2.set_ylim(0.5,1.5)
    ax2.set_xlabel("Time delay stage (ps)")
    ax2.set_ylabel("FWHM/Radius")
    ax1.grid()
    ax2.grid()
    # plt.savefig(args.output+'.png')
    plt.show()


if __name__ == "__main__":
    main()
