import sys
sys.path.append("../../utils/")
import datetime
import h5py
import numpy as np
import argparse
import utils

def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(
        description="Scales image according to the recorded current from REGAE.")
    parser.add_argument("-r", "--raw", type=str, action="store",
        help="path to the H5 raw master file")
    parser.add_argument("-i", "--input", type=str, action="store",
    help="path to the H5 input master file")
    parser.add_argument("-fi", "--file_i", type=int, action="store",
    help="file index")
    parser.add_argument("-fj", "--file_j", type=int, action="store",
        help="file index")
    parser.add_argument("-n", "--norm", type=str, action="store",
        help="path to the H5 normalization file")
    parser.add_argument("-l", "--log", type=str, action="store",
        help="current logbook")
    parser.add_argument("-o", "--output", type=str, action="store",
        help="path to the output image")
    parser.add_argument("-li", "--last_index", type=int, action="store",
        help="last index file")
    parser.add_argument("-w", "--write", type=bool, action="store",
        help="Enable writting mode in output file")

    args = parser.parse_args(raw_args)

    with h5py.File(args.norm, "r") as f:
        timestamp=str(np.array(f['entry/instrument/detector/timestamp']))
        g=open(args.log, "r")

        data_begin=timestamp[2:-3]
        data_begin=datetime.datetime.strptime(data_begin,'%a %b %d %H:%M:%S %Y')
        line_count=0
        norm_charge=[]
        for i in g:
            line=str(i)
            time_point=line[1:27]
            if line_count>0:
                time_point=datetime.datetime.strptime(time_point,'%d. %b. %Y %H:%M:%S.%f')
                timedelta=time_point-data_begin
                if 0<timedelta.seconds<40 and  timedelta.days==0:
                    norm_charge.append(float(line[45:-1]))
            line_count+=1

    norm_charge=sum(norm_charge)/len(norm_charge)

    with h5py.File(args.raw, "r") as f:
        timestamp=str(np.array(f['entry/instrument/detector/timestamp']))
        g=open(args.log, "r")

        data_begin=timestamp[2:-3]
        data_begin=datetime.datetime.strptime(data_begin,'%a %b %d %H:%M:%S %Y')
        line_count=0
        charge=[]
        for i in g:
            line=str(i)
            time_point=line[1:27]
            
            if line_count>0:
                time_point=datetime.datetime.strptime(time_point,'%d. %b. %Y %H:%M:%S.%f')
                timedelta=time_point-data_begin
                if 0<timedelta.seconds<40 and timedelta.days==0:
                    charge.append(float(line[45:-1]))
            line_count+=1

    i=args.file_i
    j=args.file_j

    if charge==[] and i<args.last_index:
        args_list=[ '-l', f'{args.log}','-r',f'{args.raw[:66]}_{i+1}.h5', '-i', f'{args.input[:59]}_{i+1}_{j}.h5', '-fi',f'{i+1}', '-fj',f'{j}','-n', f'{args.norm}','-w','False','-o',f'{args.output[:58]}{i+1}_{j}','-li',f'{args.last_index}']
        charge=main(args_list)
    if charge==[] and i>=args.last_index:
        args_list=[ '-l', f'{args.log}','-r',f'{args.raw[:66]}_{i-1}.h5', '-i', f'{args.input[:59]}_{i-1}_{j}.h5', '-fi',f'{i-1}', '-fj',f'{j}','-n', f'{args.norm}','-w','False','-o',f'{args.output[:58]}{i-1}_{j}','-li',f'{args.last_index}']
        charge=main(args_list)

    mean_charge=sum(charge)/len(charge)
    norm_factor=mean_charge/norm_charge

    with h5py.File(args.input, "r") as f:
         data=np.array(f['sum_frames_mask'])
         norm_data=np.array(f['sum_frames_mask'])/(norm_factor)
         rad_data=np.array(f['rad_average_mask'])/(norm_factor)
         sub_data=np.array(f['rad_sub'])/(norm_factor)
         rad_x=np.array(f['rad_x'])

    peaks, half=utils.calc_fwhm(sub_data,6,threshold=0,distance=15, height=0, width=8)
    peak_px=peaks+rad_x[0]
    fwhm_over_rad=[]
    intensity=[]
    for i in range(len(peaks)):
        fwhm_over_rad.append(half[0][i]/peak_px[i])
        intensity.append(sub_data[peaks[i]])

    #write output
    if args.write is True:
        f= h5py.File(args.output+'.h5', "w")
        f.create_dataset('rad_sub',data=sub_data)
        f.create_dataset('sum_frames_mask',data=data)
        f.create_dataset('norm_sum_frames',data=norm_data)
        f.create_dataset('rad_average_mask',data=rad_data)
        f.create_dataset('rad_x',data=rad_x)
        f.create_dataset('intensity',data=intensity)
        f.create_dataset('fwhm_radius',data=fwhm_over_rad)
        f.create_dataset('fwhm',data=half[0])
        f.create_dataset('peak_position',data=peak_px)
        f.close()


    return charge


if __name__ == '__main__':
    main()
