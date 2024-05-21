#!/bin/sh

# Run rocking curve fly scan calculation in rocking_curve_fly.py
# Usage:
# ./run_rock_fly.sh my-file.h5

# Authors:
# Ana Carolina Rodrigues (2021-2024)
# Mail :ana.rodrigues@desy.de

source /gpfs/cfel/user/rodria/software/env-regae/bin/activate
INPUT=$1

DATE=20220921

PEDAL_DATE=20220921_3
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11015669
p_x0=512
p_x1=528
p_y0=775
p_y1=789
NUM=5

python3 rocking_curve_fly.py -p1 ${ROOT}/processed/calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ${ROOT}/processed/calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ${ROOT}/processed/calib/gainMaps_M283.bin -g2 ${ROOT}/processed/calib/gainMaps_M281.bin -i ${ROOT}/raw/${INPUT} -s 5289 -e 6000 -p_x0=${p_x0} -p_x1=${p_x1} -p_y0=${p_y0} -p_y1=${p_y1} -o /home/rodria/mica_6_rock_${NUM}.png
