import sys

sys.path.append("../../utils/")
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
from diffractem import proc2d
import utils
import h5py
from scipy import signal


def calc_com(img):
    """
    Calculates the center of mass of an image using diffractem function.

    Parameters
    ----------
    img: np.ndarray
        Image in which you wish to calculate the center of mass
    Returns
    ----------
    center: List[int]
        center of mass position in pixels [xc,yc]
    """
    opts = utils.opts
    ## Define the region of interest
    opts.com_xrng = 200
    opts.com_yrng = 200
    img_ct = img[
        (img.shape[0] - opts.com_yrng) // 2 : (img.shape[0] + opts.com_yrng) // 2,
        (img.shape[1] - opts.com_xrng) // 2 : (img.shape[1] + opts.com_xrng) // 2,
    ]
    com = proc2d.center_of_mass(img_ct)
    com_x = round(com[0] + (img.shape[1] - opts.com_xrng) // 2)
    com_y = round(com[1] + (img.shape[0] - opts.com_yrng) // 2)

    return [com_x, com_y]


def flip_and_calc_corr(img):
    """
    Function flips an image in both axis and calculates the center from the correlation matrix with the non-flipped image. Not tested yet!

    Parameters
    ----------
    img: np.ndarray
        Image to calculate the center of diffraction

    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """
    opts = utils.opts
    opts.com_xrng = 400
    opts.com_yrng = 400
    img_ct = img[
        (img.shape[0] - opts.com_yrng) // 2 : (img.shape[0] + opts.com_yrng) // 2,
        (img.shape[1] - opts.com_xrng) // 2 : (img.shape[1] + opts.com_xrng) // 2,
    ]
    flip_data = np.flip(np.flip(img_ct.copy(), axis=0), axis=1)
    corr_mat = signal.correlate2d(img_ct, flip_data, boundary="symm", mode="same")
    y, x = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)
    # plt.imshow(img_ct,vmax=1000)
    # plt.plot(x,y,'ro')
    # plt.show()
    center = [
        (img.shape[1] - opts.com_xrng) // 2 + x,
        (img.shape[0] - opts.com_yrng) // 2 + y,
    ]
    return center


def calc_corr_sum(img, acc):
    """
    Function calculates the center of the imagae from the correlation matrix with the accumulated image. Not tested yet!

    Parameters
    ----------
    img: np.ndarray
        Image to calculate the center of diffraction
    acc: np.ndarray
        Sum of images as reference for the center shift calculation
    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """

    opts = utils.opts
    opts.com_xrng = 400
    opts.com_yrng = 400
    img_ct = img[
        (img.shape[0] - opts.com_yrng) // 2 : (img.shape[0] + opts.com_yrng) // 2,
        (img.shape[1] - opts.com_xrng) // 2 : (img.shape[1] + opts.com_xrng) // 2,
    ]
    sum_ct = img[
        (acc.shape[0] - opts.com_yrng) // 2 : (acc.shape[0] + opts.com_yrng) // 2,
        (acc.shape[1] - opts.com_xrng) // 2 : (acc.shape[1] + opts.com_xrng) // 2,
    ]

    corr_mat = signal.correlate2d(img_ct, sum_ct, boundary="symm", mode="same")
    y, x = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)

    # plt.imshow(corr_mat)
    # plt.imshow(img_ct,vmax=1000)
    # plt.plot(x,y,'ro')
    # plt.show()
    center = [
        (img.shape[1] - opts.com_xrng) // 2 + x,
        (img.shape[0] - opts.com_yrng) // 2 + y,
    ]
    return center


def flip_and_calc_com(img):
    """
    DEPRECATED. Function calculates the center of the imagae from the center of mass shift when the image is flipped in both axis. Not tested yet!

    Parameters
    ----------
    img: np.ndarray
        Image to calculate the center of diffraction

    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """
    flip_data = np.flip(np.flip(img.copy(), axis=0), axis=1)
    com_flip = calc_com(flip_data)
    com_orig = calc_com(img)
    center = [
        com_orig[0] - (com_orig[0] - com_flip[0]) // 2,
        com_orig[1] - (com_orig[1] - com_flip[1]) // 2,
    ]
    # plt.imshow(img)
    # plt.plot(com_orig[0],com_orig[1],'ro')
    # plt.show()
    # plt.imshow(flip_data)
    # plt.plot(com_flip[0],com_flip[1],'ro')
    # plt.show()
    # plt.imshow(img)
    # plt.plot(center[0],center[1],'ro')
    # plt.show()

    print(com_orig, com_flip, center)
    return center


def bring_center_to_point(data, center, point=None, radius=None):
    """
    Function shifts data center [xc,yc] to a point [xf,yf], so after transformation [xc,yc] -> [xf,yf]

    Parameters
    ----------
    img: np.ndarray
        Image to calculate the center of diffraction
    acc: np.ndarray
        Sum of images as reference for the center shift calculation
    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """

    h = data.shape[0]
    w = data.shape[1]
    if point == None:
        ##bring to center of the image
        img_center = (int(w / 2), int(h / 2))
    else:
        img_center = point

    shift = [round(img_center[0] - center[0]), round(img_center[1] - center[1])]
    new_image = np.zeros((h, w))

    if shift[0] <= 0 and shift[1] > 0:
        # move image to the left and up
        new_image[abs(shift[1]) :, : w - abs(shift[0])] = data[
            : -abs(shift[1]), abs(shift[0]) :
        ]
    elif shift[0] > 0 and shift[1] > 0:
        # move image to the right and up
        new_image[abs(shift[1]) :, abs(shift[0]) :] = data[
            : -abs(shift[1]), : -abs(shift[0])
        ]
    elif shift[0] > 0 and shift[1] <= 0:
        # move image to the right and down
        new_image[: h - abs(shift[1]), abs(shift[0]) :] = data[
            abs(shift[1]) :, : -abs(shift[0])
        ]
    elif shift[0] <= 0 and shift[1] <= 0:
        # move image to the left and down
        new_image[: h - abs(shift[1]), : w - abs(shift[0])] = data[
            abs(shift[1]) :, abs(shift[0]) :
        ]

    return new_image, shift


def calc_center_com_par(data, initial_values=None):
    """
    DEPRECATED. Function adapted to multiprocessing that calculates the center of the imagae from the center of mass shift when the image is flipped in both axis. Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    initial_values: List[int]
        Accounts if it is the first iteration

    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """
    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # plt.imshow(data,vmax=1000)
    # plt.show()

    if initial_values == None:
        global n
        n = 0
        initial_values = [0, 0]

    center = flip_and_calc_com(data)

    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # plt.imshow(data,vmax=1000)
    # plt.plot(center[0],center[1],'ro')
    # plt.show()

    print(center)

    return center


def calc_center_cc_par(data, initial_values=None):
    """
    Function adapted to multiprocessing that calculates the center of the image from the cross-correlation coefficient with the flipped image. Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    initial_values: List[int]
        Accounts if it is the first iteration

    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """

    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # plt.imshow(data,vmax=1000)
    # plt.show()

    if initial_values == None:
        # fig, ax = plt.subplots(1,1, figsize=(10, 10))
        # plt.imshow(data,vmax=1000)
        # plt.plot(initial_values[0],initial_values[1],'ro')
        # plt.show()
        global n
        n = 0

    center = flip_and_calc_corr(data)

    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # plt.imshow(data,vmax=1000)
    # plt.plot(center[0],center[1],'ro')
    # plt.show()

    print(center)

    return center


def calc_center_cc_sum(data, acc, initial_values=None):
    """
    Function that calculates the center of the image from the cross-correlation coefficient with the summed image. Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    initial_values: List[int]
        Accounts if it is the first iteration

    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """
    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # plt.imshow(data,vmax=1000)
    # plt.show()

    if initial_values == None:
        global n
        n = 0

    center = calc_corr_sum(data, acc)

    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # plt.imshow(data,vmax=1000)
    # plt.plot(center[0],center[1],'ro')
    # plt.show()

    # print(center)

    return center


def flip_and_calc_corr_coef(data, axis=None):
    """
    DEPRECATED. Function that flips an image in an axis and calculates the cross-correlation coefficient with the flipped image. Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    axis: int
        Axis in which the image will be flipped.

    Returns
    ----------
    corr: float
        cross-correlation coefficient
    """
    flip_data = np.flip(data.copy(), axis)
    corr = np.corrcoef(data.flatten(), flip_data.flatten())[0, 1]
    return corr


def calc_center_cc_axis_lin(data, axis, initial_values=None, size=2):
    """
    Function that shifts an image and calculates the center by optimizing the cross-correlation coefficient in axis. Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    axis: int
        Axis in which the cross-correlation coefficient of the image will be calculated.
    initial_values: List[int]
        Initial coordinates to begin the search.
    size: int
        Number of pixels to move the image in both directions and search the optimum correlation point.

    Returns
    ----------
    center_axis: List[int]
        center over the axis indicated xc: axis 1 yc: axis 0
    """

    h, w = data.shape[:2]
    data = data.copy()

    cc_axis = []
    cc_nonaxis = []
    cc_both = []
    steps_lst = []
    ctr_img_lst = []
    center = [w / 2, h / 2]

    x_box = np.arange(-size, size + 1, 1)
    y_box = np.zeros(len(x_box), dtype=int)

    if initial_values != None:
        center = initial_values
        data, shift = bring_center_to_point(data, center)
    else:
        center = [w / 2, h / 2]
        point = [w / 2, h / 2]
        data, shift = bring_center_to_point(data, center, point)

    if axis == 0:
        y_box = x_box.copy()
        x_box = np.zeros(len(x_box), dtype=int)

    for i, j in zip(x_box, y_box):
        step = [i, j]
        steps_lst.append(step)
        new_origin = [center[0] + step[0], center[1] + step[1]]
        new_data, shift = bring_center_to_point(data, center, new_origin)
        ctr_img_lst.append(new_data)
        cc_axis.append(flip_and_calc_corr_coef(new_data, axis))
        cc_nonaxis.append(flip_and_calc_corr_coef(new_data, abs(1 - axis)))
        cc_both.append(flip_and_calc_corr_coef(new_data, None))

    max_index = np.argmax(cc_axis)
    point = steps_lst[max_index]
    label = ["y", "x"]
    axis_opt = label[axis]
    """
    fig, ax = plt.subplots(1,2, figsize=(20, 10))
    plt.subplot(121)
    plt.title(f'Iter {n} axis {axis_opt}')
    plt.plot(cc_axis, 'b')
    plt.plot(cc_nonaxis, 'r')
    plt.plot(cc_both, 'g')
    plt.scatter(max_index, cc_axis[max_index], color='b')
    plt.subplot(122)
    plt.imshow(ctr_img_lst[max_index], cmap='jet', origin='upper', interpolation=None, norm=color.LogNorm(0.1,100))
    plt.scatter(515,532,color='r')
    #ax[1].add_patch(plt.Circle([515+steps_lst[max_index][0],532+steps_lst[max_index][1]],2,fill=False, color='b'))
    plt.show()
    """

    center_axis = center[abs(1 - axis)] - point[abs(1 - axis)]

    return center_axis


def calc_center_corr_coef_par(data, initial_values=None, size=2):
    """
    Function adapted to multiprocessing that calculates the center of the image from the cross-correlation coefficient optimization over an axis.Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    initial_values: List[int]
        Accounts if it is the first iteration
    size: int
        Number of pixels to move the image in both directions and search the optimum correlation point.
    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """
    if initial_values == None:

        initial_values = calc_com(data)
        global n
        n = 0
        ##mask for cc
        data[np.where(mask == 0)] = 0

    ref = initial_values.copy()
    x0 = calc_center_cc_axis_lin(data, 1, initial_values=initial_values, size=size)
    y0 = calc_center_cc_axis_lin(data, 0, initial_values=initial_values, size=size)

    return [x0, y0]


def calc_corr_coef_sum(data, acc):
    """
    Function that calculates the cross-correlation coefficient of an image and the summed image. Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    acc: np.ndarray
        Summed image as reference to the correlation coefficient calculation.

    Returns
    ----------
    corr: float
        correlation coefficient
    """
    corr = np.corrcoef(data.flatten(), acc.flatten())[0, 1]
    return corr


def calc_center_cc_axis_lin_sum(data, acc, axis, initial_values=None, size=2):
    """
    Function that shifts an image and calculates the center by optimizing the cross-correlation coefficient in axis (moved image and the summed image). Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    acc: np.ndarray
        Summed image as reference to the correlation coefficient calculation.
    axis: int
        Axis in which the cross-correlation coefficient of the image will be calculated.
    initial_values: List[int]
        Initial coordinates to begin the search.
    size: int
        Number of pixels to move the image in both directions and search the optimum correlation point.

    Returns
    ----------
    center_axis: List[int]
        center over the axis indicated xc: axis 1 yc: axis 0
    """
    h, w = data.shape[:2]
    data = data.copy()

    cc_axis = []
    cc_nonaxis = []
    cc_both = []
    steps_lst = []
    ctr_img_lst = []
    center = [w / 2, h / 2]

    x_box = np.arange(-size, size + 1, 1)
    y_box = np.zeros(len(x_box), dtype=int)

    if initial_values != None:
        center = initial_values
        data, shift = bring_center_to_point(data, center)
    else:
        center = [w / 2, h / 2]
        point = [w / 2, h / 2]
        data, shift = bring_center_to_point(data, center, point)

    if axis == 0:
        y_box = x_box.copy()
        x_box = np.zeros(len(x_box), dtype=int)

    for i, j in zip(x_box, y_box):
        step = [i, j]
        steps_lst.append(step)
        new_origin = [center[0] + step[0], center[1] + step[1]]
        new_data, shift = bring_center_to_point(data, center, new_origin)
        ctr_img_lst.append(new_data)
        cc_both.append(calc_corr_coef_sum(new_data, acc))

    max_index = np.argmax(cc_axis)
    point = steps_lst[max_index]
    label = ["y", "x"]
    axis_opt = label[axis]
    """
    fig, ax = plt.subplots(1,2, figsize=(20, 10))
    plt.subplot(121)
    plt.title(f'Iter {n} axis {axis_opt}')
    plt.plot(cc_axis, 'b')
    plt.plot(cc_nonaxis, 'r')
    plt.plot(cc_both, 'g')
    plt.scatter(max_index, cc_axis[max_index], color='b')
    plt.subplot(122)
    plt.imshow(ctr_img_lst[max_index], cmap='jet', origin='upper', interpolation=None, norm=color.LogNorm(0.1,100))
    plt.scatter(515,532,color='r')
    #ax[1].add_patch(plt.Circle([515+steps_lst[max_index][0],532+steps_lst[max_index][1]],2,fill=False, color='b'))
    plt.show()
    """

    center_axis = center[abs(1 - axis)] - point[abs(1 - axis)]

    return center_axis


def calc_center_corr_coef_sum_par(data, acc, initial_values=None, size=2):
    """
    Function adapted to multiprocessing that calculates the center of the image from the cross-correlation coefficient in axis (moved image and the summed image). Not tested yet!

    Parameters
    ----------
    data: np.ndarray
        Image to calculate the center of diffraction
    acc: np.ndarray
        Summed image as reference to the correlation coefficient calculation.
    initial_values: List[int]
        Accounts if it is the first iteration
    size: int
        Number of pixels to move the image in both directions and search the optimum correlation point.
    Returns
    ----------
    center: List[int]
        center  in pixels [xc,yc]
    """
    if initial_values == None:

        initial_values = calc_com(data)
        global n
        n = 0
        ##mask for cc
        data[np.where(mask == 0)] = 0

    ref = initial_values.copy()
    x0 = calc_center_cc_axis_lin_sum(
        data, acc, 1, initial_values=initial_values, size=size
    )
    y0 = calc_center_cc_axis_lin_sum(
        data, acc, 0, initial_values=initial_values, size=size
    )

    return [x0, y0]
