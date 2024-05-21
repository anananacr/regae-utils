#!/bin/sh

# Run rocking curve step scan calculation in rocking_curve_step.py
# Usage
# ./run_rock_step.sh my-files-folder/label
# Authors:
# Ana Carolina Rodrigues (2021-2024)
# Mail :ana.rodrigues@desy.de

source /gpfs/cfel/user/rodria/software/env-regae/bin/activate
INPUT=$1

DATE=20221115

PEDAL_DATE=20221115_3
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614
p_x0=557
p_x1=586
p_y0=585
p_y1=613
NUM=1


python3 rocking_curve_step.py -p1 ${ROOT}/processed/calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ${ROOT}/processed/calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ${ROOT}/processed/calib/gainMaps_M283.bin -g2 ${ROOT}/processed/calib/gainMaps_M281.bin -i ${ROOT}/raw/${INPUT} -s 100 -e 1600 -p_x0=${p_x0} -p_x1=${p_x1} -p_y0=${p_y0} -p_y1=${p_y1} -o /home/rodria/mica_5_rock_${NUM}.png
