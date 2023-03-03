import subprocess as sub
import time
import h5py
import numpy
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from numpy import exp

def gaussian(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Decode and sum up Jungfrau images.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")
    parser.add_argument("-fi", "--file_index", type=int, action="store",
        help="index file number to determine window size for peak fit")

    args = parser.parse_args(raw_args)

    #print(args.input)
    with h5py.File(args.input, "r") as f:
        data=numpy.array(f['rad_sub'])
        rad_ave_mask=numpy.array(f['rad_average_mask'])
        rad_x=numpy.array(f['rad_x'])
        offset=int(f['rad_x'][0])
        peak_pos=numpy.array(f['peak_position'])
        peak_range=numpy.array(f['peak_range'])

    fit_intensity=[]
    fit_peak_pos=[]
    fit_peak_fwhm=[]
    fit_fwhm_rad=[]
    fit_r_squared=[]
    flag=0
    
    for idx,i in enumerate(peak_pos):
        #print(abs(i-peak_range[idx][0]),abs(i-peak_range[idx][1]))
        try:
            window=int(min(abs(i-peak_range[idx][0]),abs(i-peak_range[idx][1])))
        except TypeError:
            window=3    
        
        
        try:
            x0=numpy.where(rad_x==i)[0][0]
            cut_data=data[x0-window:x0+window]
            x=numpy.arange(i-window,i+window)
            y=cut_data
            mean = sum(x * y) / sum(y)
            sigma = numpy.sqrt(sum(y * (x - mean)**2) / sum(y))
            popt,pcov = curve_fit(gaussian,x,y,p0=[max(y),mean,sigma])
        except (RuntimeError,TypeError,IndexError,ZeroDivisionError):
            flag=1
            continue
            
        residuals = y- gaussian(x, *popt)
        ss_res = numpy.sum(residuals**2)
        ss_tot = numpy.sum((y-numpy.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        #x=numpy.arange(i-window,i+window,0.05)
        #plt.plot(x,gaussian(x,*popt),'r:',label='fit')
        #x=numpy.arange(i-window,i+window)
        #plt.plot(x,cut_data)
        #print(r_squared)
        fit_intensity.append(popt[0])
        fit_peak_pos.append(popt[1])
        fwhm=popt[2]*numpy.sqrt(8*numpy.log(2))
        fit_peak_fwhm.append(fwhm)
        fit_fwhm_rad.append(fwhm/popt[1])
        fit_r_squared.append(r_squared)

        #plt.show()

    if flag==0:
        f= h5py.File(args.output+'.h5', "w")
        f.create_dataset('rad_sub',data=data)
        f.create_dataset('rad_average_mask',data=rad_ave_mask)
        f.create_dataset('rad_x',data=rad_x)
        f.create_dataset('fit_intensity',data=fit_intensity)
        f.create_dataset('fwhm_radius',data=fit_fwhm_rad)
        f.create_dataset('fwhm',data=fit_peak_fwhm)
        f.create_dataset('peak_position',data=fit_peak_pos)
        f.create_dataset('fit_r_squared',data=fit_r_squared)
        f.close()

if __name__ == '__main__':
    main()
