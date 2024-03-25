import h5py
import hdf5plugin
import numpy as np
from bblib.methods import CenterOfMass, FriedelPairs, MinimizePeakFWHM, CircleDetection
from bblib.models import PF8Info, PF8
import fabio
import matplotlib.pyplot as plt
import om.lib.geometry as geometry
import sys
from utils import azimuthal_average, gaussian
from scipy.signal import find_peaks as find_peaks
from scipy.optimize import curve_fit
from optimize_magnet import fit_gaussian
import matplotlib
matplotlib.use('Qt5Agg')

increment_current=float(sys.argv[2])

if sys.argv[3]== '-':
    height = 130
else: 
    height = int(sys.argv[3])

if sys.argv[4] == '-':
    width=4.0
else:
    width = float(sys.argv[4])

config = {
    "plots_flag": False,
	"pf8": {
		"max_num_peaks": 10000,
		"adc_threshold": 0,
		"minimum_snr": 5,
		"min_pixel_count": 2,
		"max_pixel_count": 10000,
		"local_bg_radius": 3,
		"min_res": 0,
		"max_res": 1200
		},
	"peak_region":{
		"min": 60,
		"max": 100
		},
	"canny":{
		"sigma": 3,
		"low_threshold": 0.97,
		"high_threshold": 0.99
		},	
	"bragg_peaks_positions_for_center_of_mass_calculation": 0,
	"pixels_for_mask_of_bragg_peaks": 5
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

geometry_filename="/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/rodria/geoms/JF_regae_v4_altered.geom"

PF8Config.set_geometry_from_file(geometry_filename)
data_visualize = geometry.DataVisualizer(pixel_maps=PF8Config.pixel_maps)

with h5py.File(f"{PF8Config.bad_pixel_map_filename}", "r") as f:
    mask = np.array(f[f"{PF8Config.bad_pixel_map_hdf5_path}"])

hdf5_file=sys.argv[1]
f = h5py.File(hdf5_file, "r")
shape=f["data"].shape
fwhm_over_radius=[]
peaks_pos=[]
for frame in range(shape[0]):

    plots_info={
	"file_name": f"test_{frame}",
	"folder_name": "powder",
	"root_path": "/home/rodria/bb"
    }
    data = np.array(f["/data"][frame], dtype=np.int32)
    
    PF8Config.set_geometry_from_file(geometry_filename)
    circle_detection_method = CircleDetection(
                            config=config, PF8Config=PF8Config, plots_info=plots_info
                        )
    center_coordinates_from_circle_detection = circle_detection_method(
                            data = data
                        )
    visual_data = data_visualize.visualize_data(data=data * mask)
    visual_mask = data_visualize.visualize_data(data=mask)

    bins, counts = azimuthal_average(visual_data, center_coordinates_from_circle_detection, visual_mask)
    peaks, properties = find_peaks(counts, height=height, width=width)
    x = bins[peaks[0]]
    y = counts[peaks[0]]
    peaks_pos.append(bins[peaks[0]]*75e-3)
    fit_results = fit_gaussian(bins, counts, x, right_leg=7)
    fwhm_over_radius.append(fit_results[0])

f.close()

current = np.arange(0, increment_current * (shape[0]), increment_current)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(current, peaks_pos, c="b")
ax1.set_xlabel("Current Sol67(A)")
ax1.set_ylabel("Peak radius (mm)")

ax2.scatter(current, fwhm_over_radius, c="b")
ax2.set_xlabel("Current Sol67(A)")
ax2.set_ylabel("FWHM/r")
plt.title(f"Sol67 optimization")
plt.show()
