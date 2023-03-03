import h5py
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot peak positions according to angle.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="hdf5 input image")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="hdf5 output image")
    args = parser.parse_args()
    
    size=np.arange(6,24,1)
    #pl.style.use('seaborn')
    fh, ax = plt.subplots(1,1, figsize=(8, 8))
    ax.set_facecolor('grey')
    #size=np.arange(4,86,10)
    #size=[61,60,39,38,35,34,33,32,31,30,23,22,3,2]
    size=[78,79,40,41,36,37,32,33,30,31,24,25,2,3]
    label=['-80ps','-16.65ps','-9.99ps','-3.33ps','0','9.9ps','46.7ps']
    #label=['3','5','7','9']
    n=round(len(size)/2)
    colors = pl.cm.jet(np.linspace(0.1,0.9,n))
    for idx,i in enumerate(size):
        file_path=f'{args.input}_{i}.h5'
        hf = h5py.File(file_path, 'r')
        data_name='rad_average_mask'
        rad_0 = np.array(hf[data_name])
        norm_rad_0=rad_0/np.max(rad_0)
        x=np.array(hf['rad_x'])
        #plt.plot(x,rad_0, label =f'{label[idx]}')
        if idx%2==0:
            plt.plot(x,norm_rad_0, ':',color=colors[round(idx/2)],label=f'{label[round(idx/2)]} laser off')
        else:
            print('skip')
            plt.plot(x,norm_rad_0, '.-',label =f'{label[round((idx-1)/2)]} laser on',color=colors[round((idx-1)/2)])
        hf.close()

    ax.set_xlabel('Radius [pixel]')
    ax.set_ylabel('Intensity [keV]')
    ax.grid()
    ax.set_xlim(30,180)
    plt.legend()
    plt.savefig(args.output+'.png')
    plt.show()


if __name__ == '__main__':
    main() 
