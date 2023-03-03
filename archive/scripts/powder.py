# -*- coding: utf-8 -*-
"""
Image manipulation of powder diffraction
========================================
"""
from functools import partial

import numpy as np

flip = partial(np.rot90, k=2)


def _angle_bounds(bounds):
    b1, b2 = bounds
    while b1 < 0:
        b1 += 360
    while b1 > 360:
        b1 -= 360
    while b2 < 0:
        b2 += 360
    while b2 > 360:
        b2 -= 360
    return tuple(sorted((b1, b2)))

def _trim_bounds(arr):
    """Returns the bounds which would be used in numpy.trim_zeros"""
    first = 0
    for i in arr:
        if i != 0.0:
            break
        else:
            first = first + 1
    last = len(arr)
    for i in arr[::-1]:
        if i != 0.0:
            break
        else:
            last = last - 1
    return first, last

def ellipse_average(image, center, coeff,mask=None, angular_bounds=None, trim=False):
    """
    NOT SUPPORTED! ATTEMPT TO ELLIPSE AVERAGE!
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    xc, yc = center

    # Create meshgrid and compute radial positions of the data
    # The radial positions are rounded to the nearest integer
    # TODO: interpolation? or is that too slow?
    [X, Y] = np.meshgrid(np.arange(image.shape[1])-xc  , np.arange(image.shape[0]) - yc)
    print(X,Y)
    print(X.shape,Y.shape)
    a=coeff[2]
    b=coeff[3]
    phi=coeff[-1]

    c,s=np.cos(phi), np.sin(phi)
    R = np.zeros((image.shape[0],image.shape[1]))
    el_rat=a/b
    a_rad = np.arange(10, 350,1)
    b_rad= a_rad/el_rat
    for i  in range(len(a_rad)):
        print(i)
        x,y = get_ellipse_pts([xc,yc,a_rad[i],b_rad[i],coeff[4],phi])
        for j in range(len(x)):
            R[int(y[j]),int(x[j])]=a_rad[i]
            
           
            

    fh, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(R)
    plt.show()

    intensity = []
    values=[]
    bin_size = 1

    for i in a_rad:
        mask = np.where(np.greater(R, i - bin_size) & np.less(R, i + bin_size) & (image!=0))
        values = image[mask]
        intensity.append(np.mean(values))

    return a_rad, intensity

    '''
    if angular_bounds:
        mi, ma = _angle_bounds(angular_bounds)
        angles = (
            np.rad2deg(np.arctan2(Y - yc, X - xc)) + 180
        )  # arctan2 is defined on [-pi, pi] but we want [0, pi]
        in_bounds = np.logical_and(mi <= angles, angles <= ma)
    else:
        in_bounds = np.ones_like(image, dtype=bool)

    valid = mask[in_bounds]
    image = image[in_bounds]
    Rint = Rint[in_bounds]

    px_bin = np.bincount(Rint, weights=valid * image)
    r_bin = np.bincount(Rint, weights=valid)
    print(r_bin)
    radius = np.arange(0, r_bin.size)

    # Make sure r_bin is never 0 since it it used for division anyway
    np.maximum(r_bin, 1, out=r_bin)

    # We ignore the leading and trailing zeroes, which could be due to
    first, last = 0, -1
    #if trim:
    #    first, last = _trim_bounds(px_bin)
    return radius[first:last], px_bin[first:last] / r_bin[first:last]
    
    intensity = []
    values=[]
    bin_size = 1

    for i in a_rad:
        mask = np.where(np.greater(R, i - bin_size) & np.less(R, i + bin_size) & (bad_px_mask==1))
        values = data[mask]
        intensity.append(np.mean(values))

    return a_rad, intensity
    '''
