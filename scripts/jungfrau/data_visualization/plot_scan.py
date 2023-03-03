import h5py
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
    fig, ax = plt.subplots(1,2, figsize=(20, 10))

    size=np.flip(np.arange(1,82,2))
    #start=-1*(230.5-227.5)*6.66
    #finish=(227.5-220)*6.66+0.5
    #label=np.arange(start,finish,3.33)
    label=np.arange(0,83,2)
    print(label)

    data_name='intensity'
    data_name_0='fwhm_radius'
    color=['r','g','b','orange','magenta','purple']
    count=0
    
    for idx,i in enumerate(size):
        print(label[idx],i)
        x=[label[idx],label[idx],label[idx],label[idx],label[idx],label[idx]]

        file_path=f'{args.input}_{i}.h5'
        hf = h5py.File(file_path, 'r')
        rad_0 = np.array(hf[data_name])
        rad_1 = np.array(hf[data_name_0])

        if count==0:
            print('hey')
            #norm_rad_0=rad_0.copy()
            #norm_rad_1=rad_1.copy()
            norm_rad_0=[1,1,1]
            norm_rad_1=[1,1,1]
        if i!=5:
            ax1=plt.subplot(121)
            for j in range(1,3):
                #print(rad_0[j], norm_rad_0)
                norm_int=round((rad_0[j]/norm_rad_0[j]),6)

                if count==0:
                    ax1.scatter(x[j],norm_int, color=color[j],label =f'peak_{j}',marker='o')
                else:
                    ax1.scatter(x[j],norm_int, color=color[j],marker='o')
            ax2=plt.subplot(122)
            for j in range(1,3):
            #print(rad_1[j], norm_rad_1)
                norm_fwhm=round((rad_1[j]/norm_rad_1[j]),6)
                if count==0:
                    ax2.scatter(x[j],norm_fwhm, color=color[j],label=f'peak_{j}',marker='o')
                else:
                    ax2.scatter(x[j],norm_fwhm, color=color[j],marker='o')
        hf.close()
        count+=1
    

    ax1.legend(labels=['peak_0:40','peak_1:64'])
    #ax2.legend()
    ax1.set_xlabel('Time delay (ps)')
    ax1.set_ylabel('Normal. intensity')
    #ax1.set_ylim(0.8,1.2)
    #ax2.set_ylim(0.9,1.1)
    ax2.set_xlabel('Time delay stage (ps)')
    ax2.set_ylabel('FWHM/Radius')
    ax1.grid()
    ax2.grid()
    plt.savefig(args.output+'.png')
    plt.show()


if __name__ == '__main__':
    main() 
