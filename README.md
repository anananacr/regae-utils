# regae-utils

Python library for data analysis from REGAE - Deutsches Elektronen-Synchrotron (DESY).

## Authors:

Ana Carolina Rodrigues

Mail: ana.rodrigues@desy.de

## Dependencies:

* Python 3.6.8
* requirements.txt

## Presentations
* [2023 DESY Photon Science Users' Meeting](https://docs.google.com/presentation/d/1S-YqJeze92365XabdoEd3j7OTx5JPxzq/edit?usp=share_link&ouid=114932358786595754679&rtpof=true&sd=true)
* [31st Annual Meeting of the German Crystallographic Society (DGK)](https://drive.google.com/file/d/1E2R4qOpr187P8h0Y6hYb5KgN8hUr4lbU/view?usp=share_link)

## JF1M conversion:
On jungfrau/conversion directory
### Convert and merge dark files
./convert_pedestals.sh folder_on_raw/ed_rot_scantype_00*/file_label scantype
Example:
./convert_pedestals.sh 231023_mos_c3_ms_004/ed_rot_step_001/231023_mos_c3_ms_004_001 step

### Convert images


sbatch convert_step.sh folder_on_raw/ed_rot_scantype_00*/file_label scantype start_file_index end_file_index
Example:
sbatch convert_step.sh 231023_membran_back/ed_rot_step_001/231023_membran_back_001 step 0 1000

### Assemble images

./save_assembled_images.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/folder_on_raw/ed_rot_scantype_00*/file_label -g /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/yefanov/geom/JF_regae_v4.geom -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/assembled/folder_on_raw/ed_rot_scantype_00*
Example:
./save_assembled_images.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/231019_mos_c3_ms_001/ed_rot_step_001/231019_mos_c3_ms_001_001 -g /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/yefanov/geom/JF_regae_v4.geom -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/assembled/231019_mos_c3_ms_001/ed_rot_step_001 

## Source code structure:

```bash

scripts
├── distortions
│   ├── ellipse_cut.py
│   └── ellipse.py
├── gui
│   ├── centering
│   │   ├── center_finding.py
│   │   ├── sumup_center.py
│   │   └── turbo-center-v0
│   ├── fit
│   │   ├── fit_peaks_gaus_lin.py
│   │   └── fit_peaks_gaus.py
│   ├── merge
│   │   ├── merge_average.py
│   │   ├── merge_average_radial.py
│   │   └── merge_scan.py
│   ├── processing
│   │   ├── autoproc.py
│   │   ├── convert.py
│   │   ├── rad_ave.py
│   │   ├── scan_radial_to_file.py
│   │   └── select_peak.py
│   ├── qt
│   │   ├── regae_autoproc.py
│   │   ├── results_gui.py
│   │   └── select_peaks_gui.py
│   └── scaling
│       ├── no_scaling.py
│       ├── scaling_begin.py
│       └── scaling_sum.py
├── jungfrau
│   ├── cluster-maxwell
│   │   ├── turbo_convert.sh
│   │   └── turbo_hist.sh
│   ├── conversion
│   │   ├── convert_images.py
│   │   ├── convert_pedestals.sh
│   │   ├── convert_step.sh
│   │   └── merge_all_convert.py
│   ├── data_visualization
│   │   ├── hdf_see_rings.py
│   │   ├── plot_circle.py
│   │   ├── plot_ellipse.py
│   │   ├── plot_scan_err.py
│   │   ├── plot_scan_err_ref.py
│   │   ├── plot_scan_methods.py
│   │   ├── plot_scan_mov.py
│   │   ├── plot_scan.py
│   │   ├── plot_scan_radial.py
│   │   ├── plot_scan_ring_dev.py
│   │   ├── plot_time_scan.py
│   │   └── plot_two_states.py
│   ├── histogram
│   │   └── hist_pixel.py
│   └── rocking_curve
│       ├── rocking_curve_fly.py
│       ├── rocking_curve_step.py
│       ├── run_rock_fly.sh
│       └── run_rock_step.sh
├── powder
│   ├── bib_regae_sample.yaml
│   ├── sim_pattern.py
│   └── sumup_powder.py
└── utils
    ├── calc_dist.py
    ├── check_r.py
    ├── fit_central_beam.py
    ├── label_data.py
    ├── max_res_px.py
    ├── max_res.py
    ├── shell
    │   ├── loop_center.sh
    │   └── save_average.sh
    └── utils.py

17 directories, 56 files

```
