#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=40G

#SBATCH --job-name  convert_jf
#SBATCH --output   /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/rodria/error/convert-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/rodria/error/convert-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in fly scan.
# Usage:
# sbatch convert_fly.sh 231020_mos_c3_ms_001/ed_rot_step_003/231020_mos_c3_ms_001_003 
## You need to set manually on convert_images.py how many frames you want to merge together! Use the parameter fly_frames_to_merge
# Written by Ana Carolina Rodrigues (2021-2024).
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
source /gpfs/cfel/user/rodria/software/env-regae/bin/activate

INP=$1
N=$(basename $INP)
MODE=fly
MODE_NUMBER=0
ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/

python convert_images.py -p1 ${ROOT}/processed/darks/pedal_d0_${N}_${MODE}_average.h5 -p2 ${ROOT}/processed/darks/pedal_d1_${N}_${MODE}_average.h5 -g1 ${ROOT}/processed/darks/gainMaps_M283.bin -g2 ${ROOT}/processed/darks/gainMaps_M281.bin -m ${MODE_NUMBER} -i ${ROOT}/raw/${INP} -s 0 -e 0 -o ${ROOT}/processed/converted/${INP}_master
