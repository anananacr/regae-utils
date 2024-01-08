from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import get_format, gaussian
import h5py
import math
from scipy.optimize import curve_fit

DetectorCenter = [556, 535]
Width = 8

def main():
    parser = argparse.ArgumentParser(description="Plot calculated center distribution.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    file_format = get_format(args.input)
    output_folder = args.output
    label = "center_distribution_" + args.label
    center_x = []
    center_y = []
    x_min = DetectorCenter[0] - Width
    x_max = DetectorCenter[0] + Width
    y_min = DetectorCenter[1] - Width
    y_max = DetectorCenter[1] + Width

    if file_format == "lst":
        for i in paths:

            f = h5py.File(f"{i[:-1]}", "r")
            center = np.array(f["data/refined_center"])
            hit = np.array(f["data/hit"])
            file_id = str(np.array(f["data/id"]))
            converged = np.array(f["data/converged"])

            distance = math.sqrt(
                (center[0] - DetectorCenter[0]) ** 2
                + (center[1] - DetectorCenter[1]) ** 2
            )
            if converged == 0:
                print(f"Warning!! {file_id[2:-1]} center refinement did not converged.")
            elif converged == 1:
                center_x.append(center[0])
                center_y.append(center[1])

            if distance > Width:
                print(i[:-1])
            f.close()

    print(
        f"Center refinement converged on: {len(center_x)} files of {len(paths)}.\n Convergence ratio: {round(len(center_x)*100/len(paths),1)}%"
    )

    bins = 0.1
    xedges = np.arange(x_min, x_max, bins)
    yedges = np.arange(y_min, y_max, bins)

    H, xedges, yedges = np.histogram2d(center_x, center_y, bins=(xedges, yedges))
    H = H.T

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(131, title="Detector center distribution (pixel)")
    ax.set_xlabel("Detector center in x (pixel)")
    ax.set_ylabel("Detector center in y (pixel)")
    X, Y = np.meshgrid(xedges, yedges)
    pos = ax.pcolormesh(X, Y, H, cmap="plasma")
    fig.colorbar(pos)

    ax1 = fig.add_subplot(132, title="Projection in  x (pixel)")
    proj_x = np.sum(H, axis=0)
    # bins=abs(xedges[1]-xedges[0])
    x = np.arange(xedges[0], xedges[-2], bins)
    y = proj_x

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
    residuals = y - gaussian(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    ## Calculation of FWHM
    fwhm = popt[2] * math.sqrt(8 * np.log(2))
    ## Divide by radius of the peak to get shasrpness ratio

    x_fit = np.arange(xedges[0], xedges[-1], 0.01)
    y_fit = gaussian(x_fit, *popt)
    ax1.plot(
        x_fit,
        y_fit,
        "r:",
        label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
    )

    ax1.scatter(x, proj_x, color="b")
    ax1.set_ylabel("Counts")
    ax1.set_xlabel("Detector center in x (pixel)")
    ax1.legend()

    ax = fig.add_subplot(133, title="Projection in  y (pixel)")
    proj_y = np.sum(H, axis=1)
    # bins=abs(yedges[1]-yedges[0])
    x = np.arange(yedges[0], yedges[-2], bins)
    y = proj_y

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
    residuals = y - gaussian(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    ## Calculation of FWHM
    fwhm = popt[2] * math.sqrt(8 * np.log(2))
    ## Divide by radius of the peak to get shasrpness ratio

    x_fit = np.arange(yedges[0], yedges[-1], 0.01)
    y_fit = gaussian(x_fit, *popt)

    ax.plot(
        x_fit,
        y_fit,
        "r:",
        label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
    )

    ax.scatter(x, proj_y, color="b")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Detector center in y (pixel)")
    ax.legend()
    plt.savefig(f"{args.output}/plots/{label}.png")
    plt.show()


if __name__ == "__main__":
    main()
