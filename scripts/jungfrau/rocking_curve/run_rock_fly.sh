#!/bin/sh

# Run rocking curve fly scan calculation in rocking_curve_fly.py

# ./run_rock_fly.sh my-file.h5

# Copyright � 2021-2023 Deutsches Elektronen-Synchrotron DESY,
#                       a research centre of the Helmholtz Association.
#
# Authors:
# Ana Carolina Rodrigues

#MAIL=ana.rodrigues@desy.de

# Set up environment here if necessary
INPUT=$1

DATE=20220921

PEDAL_DATE=20220921_3
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11015669
p_x0=340
p_x1=370
p_y0=380
p_y1=410

python3 rocking_curve_fly.py -p1 ${ROOT}/processed/calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ${ROOT}/processed/calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ${ROOT}/processed/calib/gainMaps_M283.bin -g2 ${ROOT}/processed/calib/gainMaps_M281.bin -i ${ROOT}/raw/${INPUT} -s 920 -e 1900 -p_x0=${p_x0} -p_x1=${p_x1} -p_y0=${p_y0} -p_y1=${p_y1}
