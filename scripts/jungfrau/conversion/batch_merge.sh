#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=4G

#SBATCH --job-name  convert_jf
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mica-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mica-%N-%j.err
 
 
source ~/scripts/regae/env-regae/bin/activate
python merge_chunks.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/240516_C3b_MICA020_3GHz_002/ed_rot_step_001/240516_C3b_MICA020_3GHz_002_001_master -s 0 -e 0 -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/240516_C3b_MICA020_3GHz_002/ed_rot_step_001/240516_C3b_MICA020_3GHz_002_001