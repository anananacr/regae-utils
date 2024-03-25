import subprocess as sub
import time
import h5py
import numpy
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from numpy import exp


def gaussian_lin(x, a, x0, sigma, m, n):
    """
    Gaussian function superimposed with linear function.

    Parameters
    ----------
    x: np.ndarray
        x array of the spectrum.
    a, x0, sigma: float
        gaussian parameters
    m,n: float
        linear parameters

    Returns
    ----------
    y: np.ndarray
        value of the function evaluated
    """
    return m * x + n + a * exp(-((x - x0) ** 2) / (2 * sigma**2))


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(description="Fit gaussian+linear peak positions.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to the H5 data master file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output image"
    )

    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:

        data = numpy.array(f["radial"])

        rad_x = numpy.array(f["radial_x"])
        offset = int(f["radial_x"][0])
        peak_pos = numpy.array(f["peak_position"])
        peak_range = numpy.array(f["peak_range"])

    fit_intensity = []
    fit_peak_pos = []
    fit_peak_fwhm = []
    fit_fwhm_rad = []
    fit_r_squared = []
    flag = 0

    for idx, i in enumerate(peak_pos):

        window = 8
        x0 = numpy.where(rad_x == i)[0][0]
        y_peak = data[x0 - window : x0 + window]
        x = numpy.arange(i - window, i + window)
        y = y_peak

        xi = numpy.where(rad_x == peak_range[idx][0])[0][0]
        xf = numpy.where(rad_x == peak_range[idx][1])[0][0]

        try:
            y_lin = numpy.hstack((data[xi:xi], data[xf:xf]))
            x_lin = numpy.hstack(
                (
                    numpy.arange(peak_range[idx][0], peak_range[idx][0]),
                    numpy.arange(peak_range[idx][1], peak_range[idx][1]),
                )
            )

            a = numpy.ones((len(x_lin), 2))
            for idx, i in enumerate(x_lin):
                a[idx][0] = i

            rough_fit = numpy.linalg.lstsq(a, y_lin, rcond=None)
            m0, n0 = rough_fit[0]

            x = numpy.arange(rad_x[xi], rad_x[xf])
            y = m0 * x + n0
            rad_sub_cut = data[xi:xf] - y
            if m0 == 0.0 and n0 == 0.0:
                rad_sub_cut -= rad_sub_cut[0] / 2 + rad_sub_cut[1] / 2
                rad_sub_cut[numpy.where(rad_sub_cut < 0)] = 0

            mean = sum(x * rad_sub_cut) / sum(rad_sub_cut)
            sigma = numpy.sqrt(sum(rad_sub_cut * (x - mean) ** 2) / sum(rad_sub_cut))

            x = rad_x[xi:xf]
            y = data[xi:xf] - (rad_sub_cut[0] / 2 + rad_sub_cut[1] / 2)
            p0 = [max(y), mean, sigma, m0, n0]
            popt, pcov = curve_fit(gaussian_lin, x, y, p0)
            residuals = y - gaussian_lin(x, *popt)
            ss_res = numpy.sum(residuals**2)
            ss_tot = numpy.sum((y - numpy.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            x_fit = numpy.arange(rad_x[xi], rad_x[xf], 0.05)

            # plt.plot(x_fit,gaussian_lin(x_fit,*popt),'r:',label='fit')
            # plt.plot(x,y)
            # print(r_squared)

            fit_intensity.append(popt[0])
            fit_peak_pos.append(popt[1])
            fwhm = popt[2] * numpy.sqrt(8 * numpy.log(2))
            fit_peak_fwhm.append(fwhm)
            fit_fwhm_rad.append(fwhm / popt[1])
            fit_r_squared.append(r_squared)
            # plt.show()

        except (RuntimeError, TypeError, IndexError, ZeroDivisionError):
            flag = 1
            # print('flag')
            continue

    if flag == 0:
        f = h5py.File(args.output + ".h5", "w")
        f.create_dataset("rad_average_mask", data=data)
        f.create_dataset("rad_x", data=rad_x)
        f.create_dataset("fit_intensity", data=fit_intensity)
        f.create_dataset("fwhm_radius", data=fit_fwhm_rad)
        f.create_dataset("fwhm", data=fit_peak_fwhm)
        f.create_dataset("peak_position", data=fit_peak_pos)
        f.create_dataset("fit_r_squared", data=fit_r_squared)
        f.close()


if __name__ == "__main__":
    main()
