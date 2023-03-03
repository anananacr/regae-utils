#!/bin/bash
#SBATCH --partition=all,cfel,maxwell
#SBATCH --time=00:20:00                           # Maximum time requested
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --chdir=$PWD        # directory must already exist!
#SBATCH --job-name=merge_regae
#SBATCH --output=om-%N-%j.out               # File to which STDOUT will be written
#SBATCH --error=om-%N-%j.err                # File to which STDERR will be written
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ana.rodrigues@cfel.de            # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge

INDEX=1

python3 merge_summed.py -i ../proc/centered/20220404_ni_off_${INDEX}_ -o  ../proc/merged/20220404_ni_off_${INDEX}_merged

exit
