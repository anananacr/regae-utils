import h5py
import argparse
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

def linear(x,m,n):
    return m*x +n

def main(raw_args=None):
    # create an axis
    parser = argparse.ArgumentParser(
    description="Calculate virtual distance from peak positions list, using reflections index and unit cell parameter for Au powder patterns.")
    parser.add_argument("-i", "--input", type=str, action="store",
    help="hdf5 input image in which peak positions is stored")
        
    args = parser.parse_args(raw_args)

    
    beam_energy=5.86 *1e-13
    px_size=75*1e-6
    _lambda=1e10*constants.h * constants.c / math.sqrt((beam_energy)**2+(2* beam_energy * constants.electron_mass * (constants.c**2)))
    print(_lambda)
    #reflections_list=[[1,1,1],[0,0,2],[0,2,2],[1,1,3],[2,2,2],[0,0,4],[1,3,3],[0,2,4]]
    reflections_list=[[1,1,1],[0,0,2],[0,2,2],[1,1,3]]
    a=407.3*1e-2
    
    q_t=[]
    for idx,i in enumerate(reflections_list):
        q_hkl=math.sqrt(i[0]**2+i[1]**2+i[2]**2)/a
        q_t.append(q_hkl)
    theta_t=[2*np.arcsin(k*_lambda/2) for k in q_t]
    print('q_t',q_t)
    print('theta_t',theta_t)

    peak_input=[90,105,146,172]   #Uncomment next line for peak positions from input file
    

    #x_o=np.array(hf['peak_position'])*px_size
    x_o=np.array(peak_input)*px_size
    
    tan_theta_o=np.tan(theta_t)

    x=tan_theta_o
    print('tan_o',x)
    a=np.ones((len(x),2))
    for idx,i in enumerate(x):
        a[idx][0]=i

    y=x_o
    rough_fit = np.linalg.lstsq(a,y, rcond=None)
    m0,n0=rough_fit[0]
    popt,pcov = curve_fit(linear,x,y,p0=[m0,0])
    residuals = y- linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('d',popt)
    print('R²',r_squared)    
    x_fit=np.arange(0.0,0.004,0.0005)
    plt.scatter(x,y,marker='o')
    plt.plot(x_fit,linear(x_fit,*popt),'r:',label='fit')

    plt.xlabel("tanθ (rad)")
    plt.ylabel("Peak radius (m)")
    plt.show()

if __name__ == '__main__':
    main()
