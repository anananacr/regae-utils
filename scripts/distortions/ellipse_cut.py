import sys
sys.path.append('../utils/')

import matplotlib.colors as color
import h5py
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
from ellipse import fit_ellipse, cart_to_pol,get_ellipse_pts


def polar_to_cart_lst(rads_lst,angs_lst):
    """
    Polar coordinates to cartesian coordinates.

    Parameters
    ----------
    rads_lst: List[int]
        radial positions in pixels
    angs_lst: List[float]
        angular positions in degrees

    Returns
    ----------
    x_lst: List[int]
        x-coordinates list
    y_lst: List[int]
        y-coordinates list
    """
    x_lst=[]
    y_lst=[]
    print(angs_lst)
    for i in range(len(rads_lst)):
        c, s = np.cos((np.pi*angs_lst[i])/180), np.sin((np.pi*angs_lst[i])/180)
        x_lst.append(int(rads_lst[i]*c+515))
        y_lst.append(int(rads_lst[i]*s+532))
    return x_lst,y_lst

def cart_to_polar_lst(x_lst,y_lst):
    """
    Cartesian to polar coordinates.

    Parameters
    ----------
    x_lst: List[int]
        x-coordinates list
    y_lst: List[int]
        y-coordinates list
    Returns
    ----------
    rads_lst: List[int]
        radial positions in pixels
    angs_lst: List[float]
        angular positions in degrees
    
    """
    rads_lst=[]
    angs_lst=[]
    
    for i in range(len(x_lst)):
        rads_lst.append(int(((x_lst[i]-515)**2+(y_lst[i]-532)**2)**0.5))
        theta=np.rad2deg(np.arctan2(y_lst[i] - 532, x_lst[i] - 515)) + 180
        angs_lst.append(theta)
    return rads_lst,angs_lst

def str_to_int_list(name):
    """
    Transform string of multiple integers separated by ',' and put into a list of ints.

    Parameters
    ----------
    name: str
        string of multiple integers separated by ','
    Returns
    ----------
    input_list: List[int]
        
    
    """
    input_lst=[]
    acc=''
    for i in range(len(name)):
        if name[i]==' ' or name[i]==',':
            if acc[0]==' ' or acc[0]==',':
                input_lst.append(int(acc[1:]))
            else:
                input_lst.append(int(acc[:]))
            acc=''
        else:
            acc+=name[i]
    print(input_lst)
    return input_lst


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot ellipse peaks position approximated.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="hdf5 image input list")
    parser.add_argument("-r", "--rings", type=str, action="store",
        help="resolution rings in pixels")
    parser.add_argument("-b", "--bins", type=str, action="store",
        help="bin size of each resolution ring in pixels")
    
    args = parser.parse_args()

    file_path=f'{args.input}'
    hf = h5py.File(file_path, 'r')
    #data_name='sum_frames_mask'
    data_name='centered_data'
    data = np.array(hf[data_name][3])
    rings=str_to_int_list(args.rings)
    bins=str_to_int_list(args.bins)
    for idx,i in enumerate(rings):
        mask=utils.ring_mask(data,i,bins[idx])
        data[~mask]=0
        positions=np.where(data>data.max()*0.9)
        rads,angs=cart_to_polar_lst(positions[1],positions[0])
        intensities=data[np.where(data>data.max()*0.90)]
        coeffs = fit_ellipse(positions[1],positions[0])
        print('Fitted parameters:')
        print('a, b, c, d, e, f =', coeffs)
        x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
        print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        ax1=plt.subplot(121)
        plt.scatter(angs,rads)
        ax2=plt.subplot(122)
        ax2.imshow(data, cmap='jet', origin='upper', interpolation=None, norm=color.LogNorm(0.1,1e4))
        x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))
        plt.plot(x, y,color='magenta')
        plt.show()

if __name__ == '__main__':
    main()    

