#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=8G

#SBATCH --job-name  convert_jf
#SBATCH --output /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mos_3-%N-%j.out
#SBATCH --error /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mos_3-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh 

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

INPUT=231019_mos_c3_ms_001/ed_rot_step_001/231020_mos_c3_ms_001_001
PEDAL=231019_mos_c3_ms_001_average
START=0
END=1200
ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted

python3 convert_images.py -p1 ${ROOT}/../darks/pedal_d0_${PEDAL}.h5 -p2 ${ROOT}/../darks/pedal_d1_${PEDAL}.h5 -g1 ${ROOT}/../darks/gainMaps_M283.bin -g2 ${ROOT}/../darks/gainMaps_M281.bin -i ${ROOT}/../../raw/${INPUT} -s ${START} -e ${END} -o ${ROOT}/${INPUT}_master;