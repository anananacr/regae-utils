import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import h5py

def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Find peak positions according to peak range.")
    parser.add_argument("-i", "--input", type=str, action="store",help="path to the H5 data master file")
    args = parser.parse_args(raw_args)

    f= h5py.File(args.input, "r")
    x=np.array(f['peak_position'])
    f.close()

    x_t=[44.31265697476193, 51.16785002002618, 72.36227398523921, 84.85229332159953]
    
    #a=np.linspace(0,2,1000)
    #b=np.linspace(-50,50,1000)
    a=np.linspace(1.06,1.09,1000)
    b=np.linspace(-1,1,1000)
    z=np.ndarray((len(a),len(b)))

    for idx,i in enumerate(a):
         for idy,j in enumerate(b):
             x0=i*x+j
             distance=math.sqrt((x0[0]-x_t[0])**2+(x0[1]-x_t[1])**2+(x0[2]-x_t[2])**2+(x0[3]-x_t[3])**2)
             z[idx,idy]=distance
    index=np.where(z==np.amin(z))
    print(a[index[0]],b[index[1]],z[index])
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.imshow(z, cmap='jet', origin='lower', interpolation=None,extent=[b[0],b[-1],a[0],a[-1]],aspect='auto')
    ax.set_ylabel('fudge factor a')
    ax.set_xlabel('x shift factor b')
    plt.show()             

if __name__ == '__main__':
    main()
