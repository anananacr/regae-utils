import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import glob
import h5py


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Find peak positions according to peak range.")
    parser.add_argument("-i", "--input", type=str, action="store",help="path to the H5 data master file")
    args = parser.parse_args(raw_args)
    fig= plt.figure(figsize=(10, 10))
    ax= fig.add_subplot(111,projection='3d')

    files=list(glob.glob(f"{args.input}*.h5"))
    x_t=[44.31265697476193, 51.16785002002618, 72.36227398523921, 84.85229332159953]
    minimization_result=[]

    for i in files:
        f=h5py.File(i, "r")
        print(i)
        x=np.array(f['peak_position'])
        f.close()
        a=np.linspace(0,2,1000)
        b=np.linspace(-50,50,1000)
        #a=np.linspace(0.9,1.1,100)
        #b=np.linspace(-1,1,100)
        z=np.ndarray((len(a),len(b)))
        

        for idx,i in enumerate(a):
            for idy,j in enumerate(b):
                x0=i*x+j
                distance=math.sqrt((x0[0]-x_t[0])**2+(x0[1]-x_t[1])**2+(x0[2]-x_t[2])**2+(x0[3]-x_t[3])**2)
                z[idx,idy]=distance
        index=np.where(z==np.amin(z))
        minimization_result.append([a[index[0]],b[index[1]],z[index]])
        ax.scatter(a[index[0]],b[index[1]],z[index],color='r',marker='o')
    print(len(minimization_result))
    ax.set_xlabel('fudge factor a')
    ax.set_ylabel('x shift factor b')
    plt.show()             

if __name__ == '__main__':
    main()
