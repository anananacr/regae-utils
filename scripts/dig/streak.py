from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
sys.path.append("/home/rodria/scripts/regae/regae-utils/scripts/jungfrau/conversion")
from apply_geom import apply_geom
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import h5py

region_label=3

def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of diffraction patterns fro MHz beam sweeping serial crystallography."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )
    parser.add_argument(
        "-m", "--mask", type=str, action="store", help="path to output data files"
    )
    parser.add_argument(
        "-g", "--geom", type=str, action="store", help="path to output data files"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    
    
    with h5py.File(f"{args.mask}", "r") as f:
        mask = apply_geom(np.array(f['data/data']),args.geom)
    
    data = np.array(Image.open(args.input))
    data[np.where(data<0)]=0
    masked_data=data*mask
    cut_data=np.array((250,30))
    cut_data=masked_data[300:550,625:655]
    plt.imshow(cut_data)
    plt.show()

    collect_positions={"y":[],"counts":[]}
    

    for idy,i in enumerate(cut_data):
        for idx, j in enumerate(i):
            collect_positions["y"].append(idy)
            collect_positions["counts"].append(j)
    
    df = pd.DataFrame.from_dict(collect_positions, dtype=float)
    print(df.y.max(), df.counts.max())
    
    bins = 2
    yedges = np.arange(0, 1.4*df.counts.max(), bins)
    xedges = np.arange(0, 250, 1)

    H, xedges, yedges = np.histogram2d(df.y, df.counts, bins=(xedges, yedges))
    H = H.T

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, title="Streak camera")
    ax.set_xlabel("y (px)")
    ax.set_ylabel("Counts (ADU)")
    X, Y = np.meshgrid(xedges, yedges)
    pos = ax.pcolormesh(X, Y, H, cmap="plasma", vmax=3)
    fig.colorbar(pos)
    
    fig.savefig(f'{args.output}/mica_region_{region_label}-long.png')
    plt.show()
            

if __name__ == "__main__":
    main()
