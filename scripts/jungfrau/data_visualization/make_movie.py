import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as color
import imageio
import subprocess as sub
from PIL import Image
import os
import cv2

center = [591, 531]
frames_per_step=1
frequency=12.5
max_frames=12999


def create_frames(file_path:str, output:str):
    ## Rings radius in pixels
    rings = [10]
    ## Center of the image [xc,yc]

    frames=np.arange(0,max_frames, 1)
    sub_frames=np.arange(1)
    print(frames)

    for frame in frames:
        #for count in sub_frames:
        #hf = h5py.File(f'{file_path}_{frame}.h5', "r")
        ##HDF5_path
        #data_name = "data"
        #data = np.array(hf[data_name][count])
        #hf.close()
        data=np.array(Image.open(f"{file_path}_{frame:06}.tif"))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pos = ax.imshow(
            data,
            cmap="cividis",
            origin="upper",
            interpolation=None,
            norm=color.Normalize(0, 200),
        )
        for i in rings:
            circle = plt.Circle(center, i, fill=True, color="red", ls=":")
            ax.add_patch(circle)
        fig.colorbar(pos,shrink=1)
        plt.title(f't = {round(((frames_per_step*frame))/frequency)} s')
        #plt.savefig(f'{output}/img_{frame}_{count}.png', transparent=False, facecolor= 'white')
        plt.savefig(f'{output}/img_{frame}.png', transparent=False, facecolor= 'white')
        plt.close()

def save_gif(file_path:str, label:str):
    frames=np.arange(0,max_frames, 1)
    sub_frames=np.arange(1)
    images=[]

    for frame in frames:
        for count in sub_frames:
            try:
                #images.append(imageio.imread(f'{file_path}/img_{frame}_{count}.png'))
                images.append(imageio.imread(f'{file_path}/img_{frame}.png'))
            except:
                print(frame, count)
            
            
    imageio.mimsave(f'{file_path}/../{label}_center_average_frames.gif',
                    images,
                    duration=0.01
                    )


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot resolution rings. Parameters need to be correctly set in code."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="input image padded"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output folder images and gif"
    )
    parser.add_argument(
        "-l", "--label", type=str, action="store", help="output folder images and gif"
    )
    args = parser.parse_args()
    file_path = args.input
    output_folder=args.output
    create_frames(file_path, output_folder)
    save_gif(output_folder, args.label)
    
   
    



if __name__ == "__main__":
    main()
