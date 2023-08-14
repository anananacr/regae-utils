#!/bin/sh

#SBATCH --partition=cfel
#SBATCH --time=0-04:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=6G

#SBATCH --job-name  convert_jf
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/error/mica_5-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/error/mica_5-%N-%j.err

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

INPUT=221115_mica_5/221115_mica_5
LABEL=20221115_mica_5
DATE=20221115
PEDAL=20221115_mica_5
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted

python3 apply_geom.py -i ${ROOT}/mica_5/mica_5 -s 0 -e 9699 -g ${ROOT}/pedal/JF_regae.geom -o ${ROOT}/mica_5_pad/mica_5
