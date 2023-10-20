#!/bin/sh
#
# Convert pedestal images using om_jungfrau_dark.py script from Onda monitor.
# Remember to set correctly paths where pedestals from JUNGFRAU 1M are stored.
#
# ./convert_step.sh 

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de


INP=231020_mica_c4_m1_001/ed_rot_step_001/231020_mica_c4_m1_001_001_start_pedal
N=231020_mica_c4_m1_001_start
beamtime=11018148
OUTPUT=/asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/processed/darks

ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/$INP*_*d0_f*.h5>pedal_d0.lst
~/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d0.lst ${OUTPUT}/pedal_d0_$N.h5

ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/$INP*_*d1_f*.h5>pedal_d1.lst
~/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d1.lst ${OUTPUT}/pedal_d1_$N.h5

