import sys
sys.path.append('../utils/')

import matplotlib.colors as color
import h5py
import numpy as np
import matplotlib.pyplot as plt
import utils
from skued import powder
import argparse
from ellipse import fit_ellipse, cart_to_pol,get_ellipse_pts
from ellipse_cut import polar_to_cart_lst, cart_to_polar_lst

def str_to_float_list(name):
    """
    Transform string of multiple floats separated by ',' and put into a list of floats.

    Parameters
    ----------
    name: str
        string of multiple integers separated by ','
    Returns
    ----------
    input_list: List[float]
        
    
    """
    input_lst=[]
    acc=''
    for i in range(len(name)):
        if name[i]==' ' or name[i]==',':
            if acc[0]==' ' or acc[0]==',':
                input_lst.append(float(acc[1:]))
            else:
                input_lst.append(float(acc[:]))
            acc=''
        else:
            acc+=name[i]
    print(input_lst)
    return input_lst


def main():
    """
    NOT SUPPORTED! ATTEMPT TO ELLIPSE AVERAGE!
    """
    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot ellipse peaks position approximated.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="hdf5 image input list")
    parser.add_argument("-c", "--coef", type=str, action="store",
        help="ellipse coeficients separted by ','  ")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="output plot image")
    
    args = parser.parse_args()

    file_path=f'{args.input}'
    hf = h5py.File(file_path, 'r')
    data_name='sum_frames_mask'
    data = np.array(hf[data_name])

    coef=str_to_float_list(args.coef)
    print(coef)
    '''
    el_rat=coef[2]/coef[3]-0.04
    el_ang=coef[-1]*180/np.pi
    print(el_rat,el_ang)
    correct_powder=utils.correct_ell(data,el_rat,el_ang)
    '''
    result=powder.ellipse_average(data,center=(515,532), coeff=coef,angular_bounds=(0,360),trim=True)
    plt.plot(result[0],result[1])
    plt.show()
    '''
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax1=plt.subplot(121)
    ax1.imshow(data, cmap='jet', origin='upper', interpolation=None, norm=color.LogNorm(0.1,1e5))
    ax2=plt.subplot(122)
    ax2.imshow(correct_powder, cmap='jet', origin='upper', interpolation=None, norm=color.LogNorm(0.1,1e5))
    plt.show()
    '''
if __name__ == '__main__':
    main()    

