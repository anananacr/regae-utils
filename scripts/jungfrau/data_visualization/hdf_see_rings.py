# -*- coding: utf-8 -*-

import om.utils.crystfel_geometry as crystfel_geometry
import h5py
import numpy as np
from scipy import constants
import matplotlib.colors as color
import argparse
import matplotlib.pyplot as plt
import math


dark = None
gain = None

def main(raw_args=None):
    global dark, gain

    # argument parser
    parser = argparse.ArgumentParser(
        description="Plot H5 images together with resolution rings. Parameters need to be correctly set in code.")
    parser.add_argument("-i", "--input", type=str, action="store",
        help="path to the H5 data master file")
    parser.add_argument("-coff", "--coff", type=float, action="store",
        help="offset distance for fine tuning of detector distance, remember to set correct parameters in code")
    parser.add_argument("-m", "--mask", type=str, action="store",
        help="path to the virtual H5 mask file")
    parser.add_argument("-g", "--geom", type=str, action="store",
        help="path to the detector geometry file (crystfel format .geom)")
    args = parser.parse_args(raw_args)

    _last_pixel_size=13333.333
    _last_detector_distance=4.9
    _last_coffset=args.coff
    beam_energy_eV=3.43*1e6

    with h5py.File(args.input, "r") as f:
        x = f["data"].shape[0]
        y = f["data"].shape[1]
        data=np.array(f["data"])

    f= h5py.File(args.mask, "r")
    mask=np.array(f['data/data'])
    f.close()

    masked_data=data*mask

    geometry_filename=args.geom
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2
        * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min())))
        + 2
    )
    x_minimum: int = (
        2
        * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min())))
        + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    frame_data=crystfel_geometry.apply_geometry_to_data(masked_data,geometry)

    center=[_img_center_x,_img_center_y]

    ## Pass resolution rings in angstroms or the by the reflection index list, resolution in angstroms will be calculated from unit cell parameter (simple cubic).
    ### Gold parameters
    '''
    _resolution_rings_in_a: List[float] = [
            2.35,
            2.04,
            1.45,
            1.23
        ]
    '''
    _resolution_rings_in_index: List[float] = [
            '[111]',
            '[002]',
            '[022]',
            '[113]'
        ]
    a=4.08

    ### Silica parameters
    '''

    _resolution_rings_in_a: List[float] = [
            1.92,
            1.36,
            0.96,
            0.86,
            0.68,
            0.64,
            0.62
        ]

    _resolution_rings_in_index: List[float] = [
            '[022]',
            '[004]',
            '[044]',
            '[026]',
            '[008]',
            '[066]',
            '[048]'
            
        ]
    a=5.43
    '''

    beam_energy=beam_energy_eV*constants.e

    lambda_=constants.h * constants.c / math.sqrt((beam_energy)**2+(2* beam_energy * constants.electron_mass * (constants.c**2)))
    print('lambda_in_A',lambda_*1e10)
    
    energy_in_V=(constants.h * constants.c)/(lambda_*constants.e)
    print('energy_in_V',energy_in_V)

    _resolution_rings_in_a=[a/math.sqrt(int(i[1])**2+int(i[2])**2+int(i[3])**2) for i in _resolution_rings_in_index]

    resolution_rings_in_pix: List[float] = [1.0]
    resolution_rings_in_pix.extend(
                [
                    1.0 *
                    _last_pixel_size
                    * (_last_detector_distance  + _last_coffset)
                    * np.tan(2.0* np.arcsin(lambda_ / (2.0 * resolution * 1e-10))
                    )
                    for resolution in _resolution_rings_in_a
                ]
            )
    print('clen',_last_detector_distance,'coff',_last_coffset,resolution_rings_in_pix)


    fh, ax = plt.subplots(1,1, figsize=(20, 20))
    ax.imshow(frame_data, norm=color.LogNorm(0.01,1e5),origin='lower',cmap='jet')

    print(center)
    for idx,i in enumerate(resolution_rings_in_pix):
        circle=plt.Circle(center,i,fill=False, color='k',ls=':')
        ax.add_patch(circle)
        if idx>0:
            plt.text(center[0]-35+int(i),center[1]-35+int(i), f'{_resolution_rings_in_index[idx-1]}', color='k', fontsize=8)
            #plt.text(center[0]-45+int(i/2),center[1]-45+int(i/2), f'{_resolution_rings_in_a[idx-1]}Å', color='k')
    plt.show()

if __name__ == '__main__':
    main()
