#!/bin/sh

INPUT=$1
START=$2
END=$3
ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/processed

for i in $(seq $START 1 $END); do
    LABEL=cc_${i}
    JNAME="cc_${i}"
    NAME="cc_${i}"
    SLURMFILE="${NAME}_${INPUT}.sh"
    echo "#!/bin/sh" > $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --partition=allcpu,cfel" >> $SLURMFILE  # Set your partition here
    echo "#SBATCH --time=0-08:00:00" >> $SLURMFILE
    echo "#SBATCH --nodes=1" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --chdir   $PWD" >> $SLURMFILE
    echo "#SBATCH --job-name  $JNAME" >> $SLURMFILE
    echo "#SBATCH --requeue" >> $SLURMFILE
    echo "#SBATCH --output    $ROOT/error/${NAME}-%N-%j.out" >> $SLURMFILE
    echo "#SBATCH --error     $ROOT/error/${NAME}-%N-%j.err" >> $SLURMFILE
    echo "#SBATCH --nice=0" >> $SLURMFILE
    echo "#SBATCH --mincpus=4" >> $SLURMFILE
    echo "#SBATCH --mem=4G" >> $SLURMFILE
    echo "#SBATCH --mail-type=ALL" >> $SLURMFILE
    echo "#SBATCH --mail-user=errodriguesana@gmail.com" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "unset LD_PRELOAD" >> $SLURMFILE
    echo "source /etc/profile.d/modules.sh" >> $SLURMFILE
    echo "module purge" >> $SLURMFILE
    echo "module load maxwell python/3.7" >> $SLURMFILE
    echo "source /home/rodria/scripts/regae/env-regae/bin/activate" >> $SLURMFILE
    echo >> $SLURMFILE
    command="./find_center_friedel.py -i ${ROOT}/${INPUT}/lists/split_${INPUT}.lst${i} -m  ${ROOT}/${INPUT}/lists/mask_regae.lst -o ${ROOT}/${INPUT} -g ${ROOT}/geom/${INPUT}/JF_regae.geom;"
    echo $command >> $SLURMFILE
    echo "chmod a+rw $PWD" >> $SLURMFILE
    sbatch $SLURMFILE 
    mv $SLURMFILE ${ROOT}/shell
done