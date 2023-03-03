import h5py
import argparse
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as color

def get_index(files_key):
    return files_key.get('index')


def main(raw_args=None):

    parser = argparse.ArgumentParser(
    description="Plot heat map of radial average scans for pump and probe experiments. Not supported, deprecated.")
    parser.add_argument("-i", "--input", type=str, action="store",
    help="hdf5 input image")
        
    args = parser.parse_args(raw_args)

    files=list(glob.glob(f"{args.input}*.h5"))
    numbers=[]

    ###take only on states
    files_keys=[]

    for i in files:
         number=i[88:90]
         if number[-1]=='_':
             file_dict={'path':i,'index':int(number[:-1]),'laser_state':int(number[:-1])%2}
             files_keys.append(file_dict)
         else:
             file_dict={'path':i,'index':int(number),'laser_state':int(number)%2}
             files_keys.append(file_dict)   

    files_keys.sort(key=get_index)

    ### order files

    n_scans=0
    for i in files_keys:
        if i['laser_state']==1:
            n_scans+=1
    print(n_scans)      
    
    n_points=500
    z=np.ndarray((n_scans,500))

    x=np.arange(0,500)
    '''
    t0=227.5
    tf=240
    ti=220
    step=0.5
    delay_step=round((3.33/0.5)*float(step),2)

    start=-1*(tf-t0)*6.6666
    finish=(t0-ti)*6.6666
    y=np.arange(start,finish+0.1,delay_step)
    '''
    y=np.arange(0,n_scans)
    scan_number=0
    for idx,i in enumerate(files_keys):
        if i['laser_state']==1:
            f=h5py.File(i['path'], "r")
            rad=np.array(f['rad_average_mask'],dtype=int)
            rad_x=np.array(f['rad_x'],dtype=int)
            rad_x=rad_x[:400]
            rad=rad[:400]
            print(rad_x)
            f.close()
            for idy,j in enumerate(rad_x):
                print(idx,j)
                z[scan_number][j]=rad[idy]
            scan_number+=1
        
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.imshow(z, cmap='jet', origin='lower', interpolation=None,extent=[x[0],x[-1],y[0],y[-1]],aspect='auto',norm=color.LogNorm(0.1,1e4))
    plt.show()


if __name__ == '__main__':
    main()
