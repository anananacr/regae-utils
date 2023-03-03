import h5py
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot peak positions according to angle.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="hdf5 input image")
    parser.add_argument("-m", "--scan_mode", type=int, action="store",
        help="magnet scan 0 pump and probe 1")
    parser.add_argument("-fi", "--first_index", type=int, action="store",
        help="hdf5 input image")
    parser.add_argument("-fp", "--first_pos", type=float, action="store",
        help="hdf5 input image")
    parser.add_argument("-li", "--last_index", type=int, action="store",
        help="hdf5 input image")
    parser.add_argument("-lp", "--last_pos", type=float, action="store",
        help="hdf5 input image")
    parser.add_argument("-s", "--step", type=float, action="store",
        help="hdf5 input image")
    parser.add_argument("-t0", "--t_zero", type=float, action="store",
        help="hdf5 output image")
    parser.add_argument("-r", "--radial_sub", type=int, action="store",
        help="radial subtraction")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="output label file")
    args = parser.parse_args(raw_args)

    delay_step=round((3.33/0.5)*float(args.step),2)

    if args.scan_mode==0:
        size=np.arange(args.first_index,args.last_index+1,1)
        start=args.first_pos-args.t_zero
        finish=args.last_pos-args.t_zero
        label=np.arange(start,finish+0.1,args.step)
        #label=np.around(label,3)
        print(label)
    else:    
        if float(args.last_pos)>float(args.first_pos):
            size=np.arange(args.first_index,args.last_index+1,1)
            start=-1*(args.last_pos-args.t_zero)*6.6666
            finish=(args.t_zero-args.first_pos)*6.6666
            label=np.flip(np.arange(start,finish+0.1,delay_step))

    n=round(len(size)/2)
    colors = pl.cm.jet(np.linspace(0.1,0.9,n))
    count=0
    for i in label:
         if abs(i)<1e-3:
             label[count]=0.0
         count+=1

    f= h5py.File(args.output+'.h5', "w")
    stamps=[]
    for idx,i in enumerate(size):
        print(label)
        file_path=f'{args.input}_{i}.h5'
        hf = h5py.File(file_path, 'r')
        if args.radial_sub==1:
            data_name='rad_sub'
        else:
            data_name='rad_average_mask'
        rad_0 = np.array(hf[data_name])
        norm_rad_0=rad_0/1
        x=np.array(hf['rad_x'])
        radial_averages=list(zip(x,norm_rad_0))
        if args.scan_mode==1:
            if idx%2==0:
                f.create_dataset(f'time_scan_{round(label[round((idx)/2)],1)}_laser_off',data=radial_averages)
                stamps.append(round(label[round((idx)/2)],1))
            else:
                f.create_dataset(f'time_scan_{round(label[round((idx-1)/2)],1)}_laser_on',data=radial_averages)
        else:
            print(size, idx, label,label[idx])
            f.create_dataset(f'magnet_scan_{round(label[idx],1)}',data=radial_averages)
            stamps.append(label[idx])

        hf.close()
    f.create_dataset(f'scan_points',data=stamps)
    f.close()
    

if __name__ == '__main__':
    main() 
