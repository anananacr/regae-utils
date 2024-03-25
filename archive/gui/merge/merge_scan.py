import h5py
import numpy
import argparse
import os
import glob


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Combine scan file in a single output H5 file."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="path to H5 data files"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output H5 file"
    )

    args = parser.parse_args(raw_args)
    intensities = []
    fwhm_rad = []
    rad_averages = []
    rad_subs = []
    rad_x = []
    peak_position = []
    r_squared = []
    fwhm = []
    files = list(glob.glob(f"{args.input}*.h5"))

    for i in files:
        f = h5py.File(i, "r")
        intensities.append(numpy.array(f["/fit_intensity"]))
        r_squared.append(numpy.array(f["/fit_r_squared"]))
        fwhm_rad.append(numpy.array(f["/fwhm_radius"]))
        fwhm.append(numpy.array(f["/fwhm"]))
        rad_averages.append(numpy.array(f["/rad_average_mask"][:500]))
        rad_x.append(numpy.array(f["/rad_x"][:500]))
        peak_position.append(numpy.array(f["/peak_position"]))
        try:
            rad_subs.append(numpy.array(f["/rad_sub"][:500]))
        except:
            pass
        f.close()

    f = h5py.File(args.output + ".h5", "w")
    f.create_dataset("intensity", data=intensities)
    f.create_dataset("r_squared", data=r_squared)
    f.create_dataset("fwhm_radius", data=fwhm_rad)
    f.create_dataset("fwhm", data=fwhm)
    f.create_dataset("rad_average_mask", data=rad_averages)
    f.create_dataset("rad_sub", data=rad_subs)
    f.create_dataset("rad_x", data=rad_x)
    f.create_dataset("peak_position", data=peak_position)
    f.close()


if __name__ == "__main__":
    main()
