#!/bin/sh
#
# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh my-file.h5 label

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

INPUT=$1
LABEL=$2
DATE=20220921
PEDAL_DATE=20220921_4
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11015669

python3 convert_all.py -p1 ${ROOT}/processed/calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ${ROOT}/processed/calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ${ROOT}/processed/calib/gainMaps_M283.bin -g2 ${ROOT}/processed/calib/gainMaps_M281.bin -i ${ROOT}/raw/${INPUT} -s 0 -e 400 -o ${ROOT}/processed/220920_mica_step
