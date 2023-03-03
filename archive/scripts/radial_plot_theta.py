import h5py
import argparse
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

def transfer_matrix(j_sol):
    e_kin=3.66
    f=e_kin*(1.274+1.247*e_kin)/(0.127+j_sol**2)
    print('f(0)',e_kin*(1.274+1.247*e_kin)/(0.127))
    d_sol=6.45
    d_sample=5.5
    d_det=10.42
    d1=(d_sol-d_sample)
    d2=(d_det-d_sol)
    m=[[1-d2/f, d1*(1-d2/f)+d2],[-1/f,1-d1/f]]
    return m
def linear(x,m,n):
    return m*x +n

def main(raw_args=None):
    # create an axis
    parser = argparse.ArgumentParser(
    description="Not supported, deprecated. Radial average plot and fitting according to peak positions expected to find the calibration factor.")
    parser.add_argument("-i", "--input", type=str, action="store",
    help="hdf5 input image")
        
    args = parser.parse_args(raw_args)

    j_sol=1.2
    beam_energy=5.86 *1e-13
    px_size=75*1e-6
    _lambda=1e10*constants.h * constants.c / math.sqrt((beam_energy)**2+(2* beam_energy * constants.electron_mass * (constants.c**2)))
    
    #reflections_list=[[1,1,1],[0,0,2],[0,2,2],[1,1,3],[2,2,2],[0,0,4],[1,3,3],[0,2,4]]
    reflections_list=[[1,1,1],[0,0,2],[0,2,2],[1,1,3]]
    a=407.3*1e-2
    
    q_t=[]
    for idx,i in enumerate(reflections_list):
        q_hkl=math.sqrt(i[0]**2+i[1]**2+i[2]**2)/a
        q_t.append(q_hkl)
    theta_t=[np.arcsin(k*_lambda/2) for k in q_t]
    print('q_t',q_t)
    print('theta_t',theta_t)

    hf= h5py.File(args.input, 'r')
    data=np.array(hf['rad_average_mask'][:500])
    x=np.array(hf['rad_x'][:500])
    x_o=np.array(hf['peak_position'])*px_size

    tm=transfer_matrix(j_sol)
    print('a01',tm[0][1],'a11',tm[1][1])
    theta_o=x_o/(2*tm[0][1])

    f_cal=np.sin(theta_t)/np.sin(theta_o)
    
    #print('M',f_cal)

    q_o=2*np.sin(theta_o)/_lambda
    print('q_o',q_o)
    #print('M',q_t/q_o)
    
    theta_x_o=x*px_size/(2*tm[0][1])
    q_x_o=2*np.sin(theta_x_o)/_lambda

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    plt.scatter(q_o,q_t)

    x_lin=q_o
    a=np.ones((len(x_lin),2))
    for idx,i in enumerate(x_lin):
        a[idx][0]=i
    #print('a,',a)
    
    x=q_o
    y=q_t       
    rough_fit = np.linalg.lstsq(a,y, rcond=None)
    m0,n0=rough_fit[0]
    #print(m0)
    popt,pcov = curve_fit(linear,x,y,p0=[m0,0])
    residuals = y- linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('popt',popt)
    print('R²',r_squared)    
    x_fit=np.arange(0,1.5,0.05)
    plt.plot(x_fit,linear(x_fit,*popt),'r:',label='fit')

    plt.xlabel("q _o (A-1)")
    plt.ylabel("q _t (A-1)")
    f_cal=popt[0]

    plt.show()

    #fig, ax = plt.subplots(1,1, figsize=(10, 10))
    #plt.scatter(theta_o,theta_t)
    #plt.scatter(np.sin(theta_o),np.sin(theta_t),color='r')
    #plt.xlabel("theta_o (rad)")
    #plt.ylabel("theta_t (rad)")

    #plt.show()

    q_c=q_x_o*f_cal

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    plt.plot(q_c,data)

    plt.xlabel("q (A-1)")
    plt.show()

if __name__ == '__main__':
    main()
