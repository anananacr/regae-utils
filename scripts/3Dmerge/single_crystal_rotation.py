import h5py
import hdf5plugin
import numpy as np
from bblib.models import PF8Info, PF8
import matplotlib.pyplot as plt
import om.lib.geometry as geometry_functions
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from os.path import basename, splitext
from scipy import constants
from PIL import Image
import tifffile as tif
import math
from matplotlib import cm

azimuth_of_the_tilt_axis = 83.35*(np.pi/180)
tilt_angle = 0.1*np.pi/180
starting_angle = -60*np.pi/180

global k
global res
global clen

config = {
    "plots_flag": True,
	"pf8": {
		"max_num_peaks": 10000,
		"adc_threshold": 0,
		"minimum_snr": 5,
		"min_pixel_count": 2,
		"max_pixel_count": 10000,
		"local_bg_radius": 3,
		"min_res": 0,
		"max_res": 1200
		}
}

PF8Config=PF8Info(
        max_num_peaks=config["pf8"]["max_num_peaks"],
        adc_threshold=config["pf8"]["adc_threshold"],
        minimum_snr=config["pf8"]["minimum_snr"],
        min_pixel_count=config["pf8"]["min_pixel_count"],
        max_pixel_count=config["pf8"]["max_pixel_count"],
        local_bg_radius=config["pf8"]["local_bg_radius"],
        min_res=config["pf8"]["min_res"],
        max_res=config["pf8"]["max_res"]
    )

def get_corrected_lab_coordinates_in_reciprocal_units(fs:int, ss:int, pixel_maps:geometry_functions.TypePixelMaps) -> tuple:
    data_shape = pixel_maps["x"].shape
    peak_index_in_slab = int(round(ss)*data_shape[1])+int(round(fs))
    radius = (pixel_maps["radius"].flatten()[peak_index_in_slab]*k)/(res*clen)
    two_theta = np.arctan2(abs(radius), k)
    phi = pixel_maps["phi"].flatten()[peak_index_in_slab]
    x = k*np.sin(two_theta)*np.cos(phi)
    y = k*np.sin(two_theta)*np.sin(phi)
    z = k - k*np.cos(two_theta)
    
    return x , y ,z

def rotate_in_x(x, y, z, angle):
    rotation_matrix=[[np.cos(angle), np.sin(angle), 0],[-1*np.sin(angle), np.cos(angle), 0], [0,0,1]]
    r = [x, y, z]
    return np.matmul(rotation_matrix, r)

def rotate_in_z(x, y, z, angle):
    rotation_matrix=[[1,0,0], [0,np.cos(angle), np.sin(angle)],[0, -1*np.sin(angle), np.cos(angle)]]
    r = [x, y, z]
    return np.matmul(rotation_matrix, r)


if sys.argv[1] == '-':
    stream = sys.stdin
else:
    stream = open(sys.argv[1], 'r')
reading_geometry = False
reading_chunks = False
reading_peaks = False
max_fs = -100500
max_ss = -100500


geometry_txt=[]

for line in stream:
    if reading_chunks:
        
        if line.startswith('End of peak list'):
            reading_peaks = False
        elif line.split(" //")[0]=='Event:':
            event_number = int(line.split(" //")[-1])
        elif line.split(" = ")[0]=='header/float//entry/shots/detector_shift_x_in_mm':
            detector_shift_in_x = float(line.split(" = ")[-1])* res *1e-3
        elif line.split(" = ")[0]=='header/float//entry/shots/detector_shift_y_in_mm':
            detector_shift_in_y= float(line.split(" = ")[-1])* res *1e-3

        elif line.startswith('  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel'):
            reading_peaks = True
            PF8Config.set_geometry_from_file()
            PF8Config.update_pixel_maps(detector_shift_in_x, detector_shift_in_y)
            ## Update phi map for the shift TO FIX in bblib
            #PF8Config.pixel_maps["phi"] = np.arctan2(PF8Config.pixel_maps["y"], PF8Config.pixel_maps["x"])

        elif reading_peaks:
            fs, ss, dump, intensity = [float(i) for i in line.split()[:4]]
            #print(intensity)
            if intensity>0:
                x_lab, y_lab, z_lab = get_corrected_lab_coordinates_in_reciprocal_units(int(fs), int(ss), PF8Config.pixel_maps)
                x, y, z = rotate_in_z(x_lab, y_lab, z_lab, azimuth_of_the_tilt_axis)
                x, y, z = rotate_in_x(x, y, z, starting_angle + (event_number*tilt_angle))
                x, y, z = rotate_in_z(x, y, z, -1 *azimuth_of_the_tilt_axis)
                reciprocal_space[int(reciprocal_space_radius+10*z), int(reciprocal_space_radius+10*y),  int(reciprocal_space_radius+10*x)] += 1e-4*intensity
                reciprocal_space[int(reciprocal_space_radius-10*z), int(reciprocal_space_radius-10*y),  int(reciprocal_space_radius-10*x)] += 1e-4*intensity


    elif line.startswith('----- End geometry file -----'):
        reading_geometry = False
        reciprocal_space_radius=int((max(PF8Config.get_detector_center())*k)/(res*clen))*10
        reciprocal_space_dimension=2*reciprocal_space_radius
        reciprocal_space = np.zeros((reciprocal_space_dimension+1, reciprocal_space_dimension+1, reciprocal_space_dimension+1), dtype=np.int16)

    elif reading_geometry:
        
        if line.split(' = ')[0]=="res":
            res=float(line.split(' = ')[-1])
        if line.split(' = ')[0]=="clen":
            clen=float(line.split(' = ')[-1].split(";")[0])
            #clen=4.95
        #elif line.split(' = ')[0]=="photon_energy":
            #beam_energy=int(line.split(' = ')[-1].split(";")[0])
        #    beam_energy = 3488009*constants.e
        #    k = math.sqrt((beam_energy)**2+(2* beam_energy * constants.electron_mass * (constants.c**2))) / (1e9*constants.h * constants.c) 
        #    print(k)
        elif line.split(' = ')[0]=="wavelength":
            wavelength = float(line.split(' = ')[-1].split(" ")[0])
            k = 1e-9/wavelength
            print(k)
        try:
            par, val = line.split('=')
            if par.split('/')[-1].strip() == 'max_fs' and int(val) > max_fs:
                max_fs = int(val)
            elif par.split('/')[-1].strip() == 'max_ss' and int(val) > max_ss:
                max_ss = int(val)
        except ValueError:
            pass

    elif line.startswith('----- Begin geometry file -----'):
        reading_geometry = True
    elif line.startswith('----- Begin chunk -----'):
        reading_chunks = True

#f = h5py.File('/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/rodria/merge/'+splitext(basename(sys.argv[1]))[0]+'-3d.h5', 'w')
#f.create_dataset('/data/data', data=reciprocal_space)
#f.close()

tif.imwrite('/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/rodria/merge/'+splitext(basename(sys.argv[1]))[0]+'-merge-shift.tif', reciprocal_space)
