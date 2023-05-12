import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as color
import imageio

def create_frames(file_path:str, output:str):
    ## Rings radius in pixels
    rings = [2, 50]
    ## Center of the image [xc,yc]
    center = [532, 510]

    frames=np.arange(1500,9500, 50)
    print(frames)

    for frame in frames:
        hf = h5py.File(file_path, "r")
        ##HDF5_path
        data_name = "data"
        data = np.array(hf[data_name][frame])
        hf.close()
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        pos = ax.imshow(
            data,
            cmap="gray",
            origin="upper",
            interpolation=None,
            norm=color.Normalize(0, 1500),
        )

        for i in rings:
            circle = plt.Circle(center, i, fill=False, color="red", ls=":")
            ax.add_patch(circle)

        fig.colorbar(pos,shrink=1)
        plt.title(f'{round(frame*0.01,2)}°')
        plt.savefig(f'{output}img_{frame}.png', transparent=False, facecolor= 'white')
        plt.close()

def save_gif(file_path:str):
    frames=np.arange(1500,9500, 50)
    images=[]
    for frame in frames:
        images.append(imageio.imread(f'{file_path}/img_{frame}.png'))
    imageio.mimsave(f'{file_path}/mica_5.gif',
                    images,
                    fps=4
                    )

def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot resolution rings. Parameters need to be correctly set in code."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output folder images and gif"
    )
    args = parser.parse_args()
    file_path = args.input
    output_folder=args.output
    create_frames(file_path, output_folder)
    save_gif(output_folder)
   
    



if __name__ == "__main__":
    main()
