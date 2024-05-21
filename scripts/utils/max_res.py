# -*- coding: utf-8 -*-
import numpy as np
from scipy import constants
import argparse


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Calculate the correponding resolution in angstroms at the edge pixel detector for the detector distance and beam energy. It doesn't take into account relativistic corrections."
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        action="store",
        help="detector ditance in meters",
    )
    parser.add_argument(
        "-n",
        "--n_edge",
        type=int,
        action="store",
        help="number of pixels from the center to the edge of the detector",
    )
    parser.add_argument(
        "-e", "--energy", type=int, action="store", help="beam energy in eV"
    )

    args = parser.parse_args(raw_args)

    _last_pixel_size = 13333.333
    _last_detector_distance = args.distance
    _pixel_detector_edge = args.n_edge

    beam_energy = args.energy

    lambda_ = (1e10 * constants.h * constants.c / constants.e) / beam_energy

    print("lambda_in_a", lambda_)
    max_resolution_in_a = lambda_ / (
        2.0
        * np.sin(
            0.5
            * np.arctan2(
                _pixel_detector_edge, _last_detector_distance * _last_pixel_size
            )
        )
    )

    print("max_resol_in_A", max_resolution_in_a)


if __name__ == "__main__":
    main()
