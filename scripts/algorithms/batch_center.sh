#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=0-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice=128
#SBATCH --mincpus=100
#SBATCH --mem=4G

#SBATCH --job-name  cc_center
#SBATCH --output    /gpfs/cfel/user/rodria/processed/regae/2022/error/cc-%N-%j.out
#SBATCH --error     /gpfs/cfel/user/rodria/processed/regae/2022/error/cc-%N-%j.err

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

./find_center.py -i /gpfs/cfel/user/rodria/processed/regae/2022/lists/mica_6.lst -m  /gpfs/cfel/user/rodria/processed/regae/2022/lists/mask_mica.lst -o /gpfs/cfel/user/rodria/processed/regae/2022/mica_6/mica_6
