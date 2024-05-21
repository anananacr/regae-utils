# -*- coding: utf-8 -*-
import numpy as np
from scipy import constants
import argparse


def main(raw_args=None):
    global dark, gain

    # argument parser
    parser = argparse.ArgumentParser(
        description="Convert resolution ansgstroms to resolution in pixels for the detector distance and beam energy. It doesn't take into account relativistic corrections."
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        action="store",
        help="detector ditance in meters",
    )
    parser.add_argument(
        "-r", "--resolution", type=float, action="store", help="resolution in angstroms"
    )
    parser.add_argument(
        "-e", "--energy", type=int, action="store", help="beam energy in eV"
    )

    args = parser.parse_args(raw_args)

    _last_pixel_size = 13333.333
    _last_detector_distance = args.distance

    resolution_in_a = args.resolution

    beam_energy = args.energy

    lambda_ = (1e10 * constants.h * constants.c / constants.e) / beam_energy

    print("lambda_in_a", lambda_)
    max_resolution_in_px = (
        1.0
        * _last_pixel_size
        * (_last_detector_distance)
        * np.tan(2.0 * np.arcsin(lambda_ / (2.0 * resolution_in_a)))
    )

    print("max_resol_in_px", max_resolution_in_px)


if __name__ == "__main__":
    main()
