# regae-utils

Python scripts for data processing at REGAE - Deutsches Elektronen-Synchrotron (DESY).

## Authors:

Ana Carolina Rodrigues (2021 - )

Mail: ana.rodrigues@desy.de

## Dependencies:

* Python 3.10.5
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

#### Additional step when measuring by less than 0.1 degrees per step. After merging the frames in a step with last command, merge sub steps so you will have each frame corresponding to 0.1 deg rotation.

sbatch merge_sub.sh folder_on_raw/ed_rot_scantype_00*/file_label n_frames_to_merge

Example:
sbatch merge_sub.sh 231222_c3b_mica_020/ed_rot_step_001/231222_c3b_mica_020_001 20 
 

### Assemble images

python save_assembled_images.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/folder_on_raw/ed_rot_scantype_00*/file_label -g /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/yefanov/geom/JF_regae_v4.geom -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/assembled/folder_on_raw/ed_rot_scantype_00*

Example:
python save_assembled_images.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/231019_mos_c3_ms_001/ed_rot_step_001/231019_mos_c3_ms_001_001 -g /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/yefanov/geom/JF_regae_v4.geom -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/assembled/231019_mos_c3_ms_001/ed_rot_step_001 

## Data analysis

### Center refinement

./turbo_center.sh folder_on_raw/ed_rot_scantype_00* list_file_label start_index end_index

Example:
./turbo_center.sh 231020_mos_c3_ms_001/ed_rot_step_002 split_231020_mos_c3_ms_001_002_step 0 2

### Detector center distribution plots

/algorithms/

Example:
python plot_center.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/stability_measurements/processed/231130_c3b_mica020/ed_rot_step_001/lists/h5_files.lst -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/stability_measurements/processed/231130_c3b_mica020/ed_rot_step_001 -l 231130_c3b_mica020_001


Example:
python plot_center_in_time.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/stability_measurements/processed/231130_c3b_mica020/ed_rot_step_001/lists/h5_files.lst -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/stability_measurements/processed/231130_c3b_mica020/ed_rot_step_001 -l 231130_c3b_mica020_001


### Optmize Sol67 current for maximum sharpness of the diffraction pattern

Example:

python optimize_magnet_powder.py /path/to/converted/file/ current_increment min_peak_height peak_width

python optimize_magnet_powder.py /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/231221_au_ref_scan/ed_magnet_step_001/231221_au_ref_scan_001_master.h5 0.1 130 4


### Stabilize images

Example:
python stabilize_images.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/center_refinement/processed/231027_tas_c4_005/ed_rot_step_003/lists/h5_files.lst -g  /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/yefanov/geom/JF_regae_v4.geom -m /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/yefanov/mask/mask_edges.h5 -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/center_refinement/processed/refined/231027_tas_c4_005/ed_rot_step_003  &
