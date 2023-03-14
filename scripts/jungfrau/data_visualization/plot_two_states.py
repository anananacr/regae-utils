import sys

sys.path.append("../../utils")
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import resolution_rings
import argparse
import matplotlib.colors as color
from matplotlib.widgets import Slider


def resolution_rings_double(data, data_2, center, rings, Imax=1000, log=True):
    """
    Plot two diffraction images and draw resolution rings in pixels according to the center passed.

    Parameters
    ----------
    data: np.ndarray
        First image state
    data_2: np.ndarray
        Second image state
    center: List[int]
        Rings center [xc,yc]
    Imax: float
        Intensity for colormap
    log: bool
        Choose between log or linear color scale.
    Returns
    ----------
    void
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax1 = plt.subplot(121)
    if log == True:
        pos1 = ax1.imshow(
            data,
            cmap="jet",
            origin="upper",
            interpolation=None,
            norm=color.LogNorm(0.1, Imax),
        )
    else:
        pos1 = ax1.imshow(
            data,
            cmap="jet",
            origin="upper",
            interpolation=None,
            norm=color.Normalize(0.0, Imax),
        )
    for i in rings:
        circle = plt.Circle(center, i, fill=False, color="magenta")
        ax1.add_patch(circle)

    ax2 = plt.subplot(122)
    if log == True:
        pos2 = ax2.imshow(
            data_2,
            cmap="jet",
            origin="upper",
            interpolation=None,
            norm=color.LogNorm(0.1, Imax),
        )
    else:
        pos2 = ax2.imshow(
            data,
            cmap="jet",
            origin="upper",
            interpolation=None,
            norm=color.Normalize(0.0, Imax),
        )
    for i in rings:
        circle = plt.Circle(center, i, fill=False, color="magenta")
        ax2.add_patch(circle)

    axcolor = "silver"
    ax_slid = plt.axes([0.15, 0.05, 0.5, 0.03], facecolor=axcolor)
    s_slid = Slider(ax_slid, "Imax", 0.1, 10 * Imax, valinit=Imax, valfmt="%i")

    def update(val):
        Imax = round(s_slid.val)
        ax1 = plt.subplot(121)
        ax1.set_title("3")
        if log == True:
            pos1 = ax1.imshow(
                data,
                cmap="jet",
                origin="upper",
                interpolation=None,
                norm=color.LogNorm(0.1, Imax),
            )
        else:
            pos1 = ax1.imshow(
                data,
                cmap="jet",
                origin="upper",
                interpolation=None,
                norm=color.Normalize(0.0, Imax),
            )

        for i in rings:
            circle = plt.Circle(center, i, fill=False, color="magenta", ls=":")
            ax1.add_patch(circle)

        ax2 = plt.subplot(122)
        ax2.set_title("85")
        if log == True:
            pos2 = ax2.imshow(
                data_2,
                cmap="jet",
                origin="upper",
                interpolation=None,
                norm=color.LogNorm(0.1, Imax),
            )
        else:
            pos2 = ax2.imshow(
                data,
                cmap="jet",
                origin="upper",
                interpolation=None,
                norm=color.LogNorm(0.0, Imax),
            )
        for i in rings:
            circle = plt.Circle(center, i, fill=False, color="magenta", ls=":")
            ax2.add_patch(circle)

        fig.canvas.draw()

    s_slid.on_changed(update)
    plt.show()


def main():

    # argument parser
    parser = argparse.ArgumentParser(description="Plot resolution rings.")
    parser.add_argument(
        "-i1", "--input1", type=str, action="store", help="hdf5 input image first stage"
    )
    parser.add_argument(
        "-i2",
        "--input2",
        type=str,
        action="store",
        help="hdf5 input image second stage",
    )
    parser.add_argument("-l", "--log", type=bool, action="store", help="color scale")

    args = parser.parse_args()

    file_path = args.input1
    hf = h5py.File(file_path, "r")
    ## Set hdf5 data path
    data_name = "sum_frames_mask"
    data = np.array(hf[data_name])
    hf.close()

    file_path = args.input2
    hf = h5py.File(file_path, "r")
    ## Set hdf5 data path
    data_name = "sum_frames_mask"
    data_2 = np.array(hf[data_name])
    hf.close()

    ## Set resolution rings in pixels and center of rings
    rings = [45, 65, 75, 99, 111]
    center = [515, 532]
    ## Change Imax for data intensity level
    resolution_rings_double(data, data_2, center, rings, Imax=1e4, log=args.log)


if __name__ == "__main__":
    main()
