import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy import sparse
import scipy.signal as signal
from matplotlib.widgets import Slider
from scipy.sparse.linalg import spsolve
import sys
from skued import azimuthal_average


def ring_mask(data, rad, bin=1):
    """
    Make a  ring mask for the data

    Parameters
    ----------
    data: np.ndarray
        Image in which mask will be shaped
    rad: int
        Outer radius of the mask
    bin: int
        Bin size of the ring
    Returns
    ----------
    mask: np.ndarray
    """

    bin_size = bin
    a = data.shape[0]
    b = data.shape[1]
    center = [b / 2, a / 2]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    return np.greater(R, rad - bin_size) & np.less(R, rad + bin_size) & (data != 0)


def radial_average(data, bad_px_mask):
    """
    Calculate azimuthal integration of data in relation to the center of the image

    Parameters
    ----------
    data: np.ndarray
        Image in which mask will be shaped
    bad_px_mask: np.ndarray
        Bad pixels mask to not be considered in the calculation
    Returns
    ----------
    rad: List[int]
        x axis radius in pixels

    intensity: List[float]
        Integrated intensity
    """
    a = data.shape[0]
    b = data.shape[1]
    if bad_px_mask is None:
        bad_px_mask = np.ones((a, b))
    center = [b / 2, a / 2]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    global R
    R = np.sqrt(np.square(X) + np.square(Y))
    rad = np.arange(0, 600)
    intensity = []
    values = []
    bin_size = 1

    for i in rad:
        mask = np.where(
            np.greater(R, i - bin_size) & np.less(R, i + bin_size) & (bad_px_mask == 1)
        )
        values = data[mask]
        intensity.append(np.mean(values))

    return rad, intensity


def find_peaks(rad, avg):
    """
    Calculate peak positions for an azimuthal integration spectrum

    Parameters
    ----------
    rad: List[int]
        x axis radius in pixels

    avg: List[float]
        Integrated intensity
    Returns
    ----------
    peaks: np.ndarray
        Index of peak positions in pixels

    height: np.ndarray
        Intensity of peaks
    """

    index, height = signal.find_peaks(avg, threshold=0.001, height=-2)
    plt.plot(rad, avg)
    peaks = index
    plt.scatter(peaks, height["peak_heights"], color="r")
    return peaks, height["peak_heights"]


def resolution_rings(data, center, rings, Imax=1000):
    """
    Interactive plot data showing resolution rings.

    Parameters
    ----------
    data: np.ndarray
        Diffraction image data

    center: List[int]
        Center position in which respct the rings will be plotted [xc,yc]
    rings: List[int]
        Rings radius in pixels to be plotted
    Imax:
        Level of intensity for heat map visualization

    Returns
    ----------
    void
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    pos = ax.imshow(
        data,
        cmap="jet",
        origin="upper",
        interpolation=None,
        norm=color.LogNorm(0.1, Imax),
    )

    for i in rings:
        circle = plt.Circle(center, i, fill=False, color="magenta", ls=":")
        ax.add_patch(circle)

    # fig.colorbar(pos,shrink=1)

    axcolor = "silver"
    ax_slid = plt.axes([0.15, 0.05, 0.5, 0.03], facecolor=axcolor)
    s_slid = Slider(ax_slid, "Imax", 0.1, 10 * Imax, valinit=Imax, valfmt="%i")

    def update(val):
        Imax = round(s_slid.val)
        pos = ax.imshow(
            data,
            cmap="jet",
            origin="upper",
            interpolation=None,
            norm=color.LogNorm(0.1, Imax),
        )

        for i in rings:
            circle = plt.Circle(center, i, fill=False, color="magenta", ls=":")
            ax.add_patch(circle)

        fig.canvas.draw()

    s_slid.on_changed(update)
    plt.show()


def baseline_als(y, lam, p, niter=10):
    """
    Calculate baseline using Asymmetric Least Squares (ALS)

    Parameters
    ----------
    y: List[float]
        Spectra intensity
    lam: int
        2nd derivative constraint
    p: float
        Weighting of positive residuals
    niter: int
        Maximum number of iterations

    Returns
    ----------
    z: np.ndarray()
        Fitted baseline spectra intensity
    """

    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def sum_nth(data, n, phase):
    """
    Sum of images that matches the phase from an interval o n images
    Parameters
    ----------
    data: np.ndarray
        Array of images to be summed data.shape[0] is n
    n: int
        Total number of images in data
    phase: int
        Summ only the phase-th image from data stack

    Returns
    ----------
    accumulate: np.ndarray()
         Sum of phase-th images from n images in data
    """
    n = data.shape[0]

    accumulate = np.zeros((data.shape[1], data.shape[2]))
    for idx, i in enumerate(data):
        if idx % n == phase:
            accumulate += i
    return accumulate


def calc_fwhm(data, n, threshold=0.001, height=0, width=1, distance=5):
    """
    Calculate Full width at half maximum (FWHM) in pixels from raw data. Not a fit!
    Parameters
    ----------
    data: np.ndarray
        Powder diffraction spectra
    n: int
        Calculate only for the first n peaks
    **kwargs: float
        scipy.signal.find_peaks parameters

    Returns
    ----------
    peaks: np.ndarray()
        Peaks index
    results_half: np.ndarray()
        Corresponding peaks FWHM in pixels
    """

    index, height = signal.find_peaks(data, threshold, height, width, distance)
    if n == -1:
        # take all peaks
        peaks = index[:]
    else:
        peaks = index[:n]
    results_half = signal.peak_widths(data, peaks, rel_height=0.5)
    return peaks, results_half