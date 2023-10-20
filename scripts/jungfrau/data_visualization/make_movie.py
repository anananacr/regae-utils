import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as color
import imageio

def create_frames(file_path:str, output:str):
    ## Rings radius in pixels
    rings = [10]
    ## Center of the image [xc,yc]
    center = [554, 543]

    frames=np.arange(598,1200, 1)
    sub_frames=np.arange(10)
    print(frames)

    for frame in frames:
        for count in sub_frames:
            hf = h5py.File(f'{file_path}_{frame}.h5', "r")
            ##HDF5_path
            data_name = "data"
            data = np.array(hf[data_name][count])
            hf.close()
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
            plt.title(f't = {round(((10*frame)+count)*0.08)} s')
            plt.savefig(f'{output}/img_{frame}_{count}.png', transparent=False, facecolor= 'white')
            plt.close()

def save_gif(file_path:str):
    frames=np.arange(0,1201, 1)
    sub_frames=np.arange(10)
    images=[]

    for frame in frames:
        for count in sub_frames:
            try:
                images.append(imageio.imread(f'{file_path}/img_{frame}_{count}.png'))
            except:
                print(frame, count)
            
            
    imageio.mimsave(f'{file_path}/../mos_c3_center_fast.gif',
                    images,
                    duration=20
                    )

def save_gif_fly(file_path:str):
    frames=np.arange(1000,7000, 1)
    images=[]
    for frame in frames:
        try:
            images.append(imageio.imread(f'{file_path}/img_{frame}.png'))
        except:
            print(frame)
            
    imageio.mimsave(f'{file_path}/../mica_4_1000_7000.gif',
                    images,
                    duration=0.02
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
    #create_frames(file_path, output_folder)
    save_gif(output_folder)
   
    



if __name__ == "__main__":
    main()
