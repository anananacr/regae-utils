import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from numpy import exp

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple     
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Decode and sum up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")


    args = parser.parse_args(raw_args)

    with h5py.File(args.input, "r") as f:
        cut_beam=np.array(f['sum_frames_mask'])[505:565,500:565]
    cut_beam[np.where(cut_beam<15000)]=0
    #cut_beam[np.where(cut_beam==0)]=np.nan
    #fig, ax = plt.subplots(1,1, figsize=(10, 10))
    #ax.imshow(cut_beam)
    #plt.show()
    proj_y=[]
    proj_x=[]
    for idx,i in enumerate(cut_beam[:,0]):
        proj_y.append(np.sum(cut_beam[idx,:]))
    for idy,i in enumerate(cut_beam[0,:]):
        proj_x.append(np.sum(cut_beam[:,idy]))
    print(proj_x)

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    plt.plot(proj_x)
    plt.show()

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    plt.plot(proj_y)
    plt.show()

    ##initial_guess

    x=np.arange(0,len(cut_beam[0,:]))
    y=proj_x
    mean_x0 = sum(x * y) / sum(y)
    sigma_x0 = np.sqrt(sum(y * (x - mean_x0)**2) / sum(y))

    x=np.arange(0,len(cut_beam[:,0]))
    y=proj_y
    mean_y0 = sum(x * y) / sum(y)
    sigma_y0 = np.sqrt(sum(y * (x - mean_x0)**2) / sum(y))

    theta=0
    amplitude=3e5
    offset=0

    p0=[amplitude,mean_x0,mean_y0,sigma_x0,sigma_y0,theta,offset]
    popt,pcov = curve_fit(twoD_Gaussian,cut_beam,p0)
    print(popt)


if __name__ == '__main__':
    main()
