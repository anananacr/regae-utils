#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import get_format, gaussian
import h5py
import math
from scipy.optimize import curve_fit

DetectorCenter = [554,543]
frequency=12.5
frames_per_step=10

def calculate_time_point_from_path(file_path:str, frame:int):
    #print(((file_path.split('/')[-1]).split('.')[0]).split('_')[-1])
    file_index=int(((file_path.split('/')[-1]).split('.')[0]).split('_')[-1])
    n=(frames_per_step*file_index)+frame
    return n/frequency

def main():
    parser = argparse.ArgumentParser(description="Plot calculated center distribution.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    file_format = get_format(args.input)
    output_folder = args.output
    label = "center_distribution_" + output_folder.split("/")[-1]
    g=open(f'{args.output}/plots/beam_center.csv', 'w')
    g.write('time center_x center_y\n')
    #print(label)
    center_x = []
    center_y = []
    x_min=550
    x_max=561
    y_min=538
    y_max=549
    time=[]
    if file_format == "lst":
        for i in paths[:]:
            
            f = h5py.File(f"{i[:-1]}", "r")
            center = np.array(f["refined_center"])
            file_path=str(np.array(f["id"]))
            frame=int(np.array(f["index"]))
            error=math.sqrt((center[0]-DetectorCenter[0])**2+(center[1]-DetectorCenter[1])**2)
            if center[1]>y_min and center[1]<y_max and center[0]<x_max and center[0]>x_min:
                timestamp=calculate_time_point_from_path(file_path,frame)
                time.append(timestamp)
                center_x.append(center[0])
                center_y.append(center[1])
                g.write(f'{timestamp} {center[0]} {center[1]}\n')
            #if error>10:
            #    print(i[:-1])
            #f.close()
            #except KeyError:
            #    continue
                #print(i[:-1])
            #except:
            #    print("OS", i[:-1])
    #print(len(center_x))
    g.close()
    

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(121, title='Detector center in x (pixel)')
    
    ax.set_ylabel("Detector center in x (pixel)")
    ax.set_xlabel("Time (s)")
    ax.scatter(time, center_x, marker='.', s=2)

    ax = fig.add_subplot(122, title='Detector center in y (pixel)')
    
    ax.set_ylabel("Detector center in y (pixel)")
    ax.set_xlabel("Time (s)")
    ax.scatter(time, center_y, color='orange', marker='.', s=2)

    #ax.legend()
    plt.savefig(f"{args.output}/plots/{label}_time.png")
    plt.show()

    


if __name__ == "__main__":
    main()
