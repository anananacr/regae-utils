import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy import sparse
import scipy.signal as signal
from matplotlib.widgets import Slider
from scipy.sparse.linalg import spsolve
import sys
from skued import azimuthal_average


class opts:
    """
    Set options to calculate center of mass
    """

    com_xrng = 512
    com_yrng = 512
    com_threshold = 0


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


def sum_nth_mask(data, n, phase):
    """
    Sum of mask that matches the phase from an interval o n masks
    Parameters
    ----------
    data: np.ndarray
        Array of masks to be summed data.shape[0] is n
    n: int
        Total number of images in data
    phase: int
        Summ only the phase-th masks from data stack

    Returns
    ----------
    accumulate: np.ndarray()
        Sum of phase-th masks from n images in data
    """
    n = data.shape[0]
    accumulate = np.zeros((data.shape[1], data.shape[2]))
    zeros = accumulate.copy()
    for idx, i in enumerate(data):
        if idx % n == phase:
            accumulate += i
            if idx != 0:
                accumulate = np.where(accumulate == 1, zeros, accumulate // 2)
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


def ring_deviation(data, radius):
    """
    Intensity deviation in a circular radius according to the angle
    Parameters
    ----------
    data: np.ndarray
        Powder diffraction image
    radius: int
        Ring radius

    Returns
    ----------
    intensities: np.ndarray()
        Intensities obeserved in a ring slice of the data
    deg: List[float]
        Corresponding theta positions in degrees
    """
    a = data.shape[0]
    b = data.shape[1]
    center = [b / 2, a / 2]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    intensities = []
    bin_size = 1
    mask = np.where(
        np.greater(R, radius - bin_size) & np.less(R, radius + bin_size) & (data != 0)
    )
    intensities = data[mask]
    deg = []
    for i in range(len(mask[0])):
        theta = (
            np.rad2deg(np.arctan2(mask[0][i] - center[1], mask[1][i] - center[0])) + 180
        )
        deg.append(round(theta, 2))
    return intensities, deg


def ellipse_measure(data, beam_cut=0):

    """
    Take the sliced radial average of data in few degrees, calculate the peak position for the first 4 peaks according to the angle.
    Parameters
    ----------
    data: np.ndarray
        Powder diffraction image
    beam_cut: int
        beam radial profile to be masked

    Returns
    ----------
    peak_positions: List[int]
        Peak positions in pixels
    angle: List[float]
        Corresponding theta positions in degrees
    """
    peak_positions = []
    angle = []
    intensity = []
    for i in range(15, 350, 5):
        rad_signal_masked = azimuthal_average(
            data, center=(515, 532), angular_bounds=(i - 15, i + 5), trim=True
        )
        rad_signal_cut = []
        rad_signal_cut.append(np.array(rad_signal_masked[0][beam_cut:]))
        rad_signal_cut.append(np.array(rad_signal_masked[1][beam_cut:]))
        baseline = baseline_als(rad_signal_cut[1], 1e4, 0.1)
        rad_sub = rad_signal_cut[1] - baseline
        rad_sub[np.where(rad_sub < 0)] = 0
        # plt.plot(rad_sub)
        # plt.show()

        peaks, half = calc_fwhm(
            rad_sub, 4, threshold=400, height=1, width=5, distance=8
        )
        intensity = []
        for j in range(len(peaks)):
            intensity.append(rad_sub[peaks[j]])
        # plt.plot(rad_sub)
        # plt.scatter(peaks,intensity,c='r')
        # plt.show()

        peak_px = list(peaks + rad_signal_cut[0][0])
        # print(intensity)
        if peak_px == []:
            peak_px = [0, 0, 0, 0]
        if len(peak_px) == 1:
            peak_px.append(0)
            peak_px.append(0)
            peak_px.append(0)
        if len(peak_px) == 2:
            peak_px.append(0)
            peak_px.append(0)
        if len(peak_px) == 3:
            peak_px.append(0)
        if len(peak_px) != 0:
            angle.append(i + 5)
            peak_positions.append(peak_px)

    # print(peak_positions)
    # print(angle)

    return angle, peak_positions


def check_ellipticity(data, el_rat=1, el_ang=0):
    """
    Check ellipticity
    Parameters
    ----------
    data: np.ndarray
        Powder diffraction image
    el_rat: float
        ratio between ellipse principal axis#
    el_ang: float
        angle of rotation in degrees

    Returns
    ----------
    peak_positions: List[int]
        Peak positions in pixels
    angle: List[float]
        Corresponding theta positions in degrees
    """
    data = correct_ell(data.copy(), el_rat, el_ang)

    peak_positions = []
    angle = []
    intensity = []
    for i in range(19, 359, 1):
        rad_signal_masked = azimuthal_average(
            data, center=(515, 532), angular_bounds=(i - 19, i + 1), trim=True
        )
        rad_signal_cut = []
        rad_signal_cut.append(np.array(rad_signal_masked[0][10:]))
        rad_signal_cut.append(np.array(rad_signal_masked[1][10:]))
        baseline = baseline_als(rad_signal_cut[1], 1e3, 0.1)
        rad_sub = rad_signal_cut[1] - baseline
        rad_sub[np.where(rad_sub < 0)] = 0
        # plt.plot(rad_sub)
        # plt.show()

        peaks, half = calc_fwhm(
            rad_sub, -1, threshold=100, height=1, width=3, distance=5
        )
        intensity = []
        for j in range(len(peaks)):
            intensity.append(rad_sub[peaks[j]])
        # plt.plot(rad_sub)
        # plt.scatter(peaks,intensity,c='r')
        # plt.show()

        peak_px = list(peaks + rad_signal_cut[0][0])
        for j in peak_px:
            angle.append(i + 1)
            peak_positions.append(j)

        """
        if peak_px==[]:
            peak_px=[0,0,0,0]
        if len(peak_px)==1:
            peak_px.append(0)
            peak_px.append(0)
            peak_px.append(0)
        if len(peak_px)==2:
            peak_px.append(0)
            peak_px.append(0)
        if len(peak_px)==3:
            peak_px.append(0)
        if len(peak_px)!=0:
            angle.append(i+5)
            peak_positions.append(peak_px)
 
    max_len=0
    for i in peak_positions:
        if len(i)>max_len:
            max_len=len(i)
    print(max_len)
    """
    # ellipticity correction

    rad_range = (25, 100)
    powder_polar = np.histogram2d(
        peak_positions,
        angle,
        bins=[np.linspace(*rad_range, 200), np.linspace(0, 360, 360)],
    )
    corr = powder_polar[0] * np.roll(
        powder_polar[0], powder_polar[0].shape[1] // 4, axis=1
    )

    plt.close("all")
    fh, ax = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    ax[0].pcolormesh(powder_polar[2][:-1], powder_polar[1][:-1], powder_polar[0])
    ax[0].set_xlabel("Azimuth (deg)")
    ax[0].set_ylabel("Corrected radius")
    ax[1].plot(
        np.nanmean(powder_polar[0] ** 2, axis=1),
        powder_polar[1][:-1],
        label="Mean squared pattern",
    )
    ax[1].plot(
        np.nanmean(corr, axis=1), powder_polar[1][:-1], label="Quadrant correlation"
    )
    ax[1].legend()
    ax[1].set_xlabel("Mean squared counts")

    print(
        f"Median ellipticity variance: {np.nanmedian((np.nanvar(powder_polar[0], axis=1)/np.nanmean(powder_polar[0],axis=1))):.2f} \n"
        f"Rel. quadrant correlation: {np.mean(corr)/np.mean(powder_polar[0]**2):.3g}"
    )
    # plt.show()

    return angle, peak_positions


def correct_ell(data, el_rat, el_ang):
    """
    Correct ellipticity stratching  and rotating an image
    Parameters
    ----------
    data: np.ndarray
        Powder diffraction image
    el_rat: float
        ratio between ellipse principal axis#
    el_ang: float
        angle of rotation in degrees

    Returns
    ----------
    corrected_image: np.ndarray
        Corrected image for ellipticity
    """

    corrected_image = np.zeros((data.shape[:]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x1, y1 = calc_new_coord(j, i, el_rat, el_ang)
            if 0 < x1 < data.shape[1] and 0 < y1 < data.shape[0]:
                corrected_image[y1, x1] = data[i, j]
    # plt.imshow(corrected_image)
    # plt.show()
    return corrected_image


def calc_new_coord(x0, y0, el_rat, el_ang):
    """
    Coordinates transformation given the ellipse ratio and angle.

    Parameters
    ----------
    x0: int
        x axis coordinate of a point in the image
    y0: int
        y axis coordinate of a point in the image
    el_rat: float
        ratio between ellipse principal axis#
    el_ang: float
        angle of rotation in degrees

    Returns
    ----------
    x1: int
        x axis transformated coordinate of a point in the image
    y1: int
        y axis transformated coordinate of a point in the image
    """
    c, s = np.cos(np.pi / 180 * el_ang), np.sin(np.pi / 180 * el_ang)
    x1, y1 = 1 / el_rat**0.5 * (c * x0 - s * y0), el_rat**0.5 * (s * x0 + c * y0)
    x1, y1 = c * x1 + s * y1, -s * x1 + c * y1
    x1, y1 = int(x1), int(y1)

    return x1, y1
