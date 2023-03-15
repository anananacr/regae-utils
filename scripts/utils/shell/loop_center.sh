#!/bin/sh
#
# Convert images using sumup_center.py script.
# Remember to set correctly paths where data from JUNGFRAU 1M are stored.
#
# ./loop_center.sh my_file.h5 label

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

INPUT=$1
DATE=20220922
LABEL=$2
PEDAL_DATE=20220922_1

## Set here your ROOT path
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11015669

STEP=25
ALG=0

for i in $(seq 0 1 0); do
    for j in $(seq 0 ${STEP} 0); do
        python3 ../../gui/centering/sumup_center.py -p1 ${ROOT}/scratch_cc/rodria/calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ${ROOT}/scratch_cc/rodria/calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ${ROOT}/scratch_cc/rodria/calib/gainMaps_M283.bin -g2 ${ROOT}/scratch_cc/rodria/calib/gainMaps_M281.bin -i ${ROOT}/raw/${INPUT}_master_${i}.h5 -m ${ROOT}/scratch_cc/rodria/calib/mask_mica_4.h5 -f ${ALG} -n ${STEP} -b $j -g ${ROOT}/scratch_cc/rodria/calib/JF_regae_v1_mica_4.geom -o ${ROOT}/scratch_cc/rodria/processed/centered/${DATE}/${DATE}_${LABEL}_${i}_${j}; done; done;
