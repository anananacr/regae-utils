#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=40G

#SBATCH --job-name  merge_frames
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/center_refinement/error/merge-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/center_refinement/error/merge-%N-%j.err
# Usage:
#sbatch merge_sub.sh 240516_C3b_MICA020_3GHz_002/ed_rot_step_001/240516_C3b_MICA020_3GHz_002_001 20
# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module purge all
source /gpfs/cfel/user/rodria/software/env-regae/bin/activate

ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted
INPUT=$1
N=$2

python merge_sub_steps.py -i ${ROOT}/${INPUT} -n ${N} ;

