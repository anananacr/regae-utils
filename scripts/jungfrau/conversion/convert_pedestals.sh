#!/bin/sh
#
# Convert pedestal images using om_jungfrau_dark.py script from Onda monitor.
# Remember to set correctly paths where pedestals from JUNGFRAU 1M are stored.
#
# ./convert_step.sh 

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de


INP=221115_mica_6/221115_mica_6_start_pedal
N=20221115_3
beamtime=11016614

cp -r ~/scripts/calib/ /asap3/fs-bmx/gpfs/regae/2022/data/$beamtime/processed/

ls /asap3/fs-bmx/gpfs/regae/2022/data/$beamtime/raw/$INP*_*d0_f*.h5>~/scripts/inp/pedal_d0.lst
~/software/om/bin_src/om_jungfrau_dark.py ~/scripts/inp/pedal_d0.lst ~/scripts/calib/pedal_d0_$N.h5

ls /asap3/fs-bmx/gpfs/regae/2022/data/$beamtime/raw/$INP*_*d1_f*.h5>~/scripts/inp/pedal_d1.lst
~/software/om/bin_src/om_jungfrau_dark.py ~/scripts/inp/pedal_d1.lst ~/scripts/calib/pedal_d1_$N.h5

cp -r ~/scripts/calib/pedal_d1_$N.h5 /asap3/fs-bmx/gpfs/regae/2022/data/$beamtime/processed/calib/

cp -r ~/scripts/calib/pedal_d0_$N.h5 /asap3/fs-bmx/gpfs/regae/2022/data/$beamtime/processed/calib/
exit
