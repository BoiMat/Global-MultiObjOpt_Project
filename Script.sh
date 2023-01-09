#!/bin/bash
#PBS -q dssc_gpu
#PBS -l nodes=1:ppn=48
#PBS -l walltime=03:00:00
#PBS -N weak

cd $PBS_O_WORKDIR/

module load conda/4.9.2
conda activate Global

start=`date +%s`

python test.py

end=`date +%s`

runtime=$((end-start))

minutes=$(( (runtime % 3600) / 60 )) 
seconds=$(( (runtime % 3600) % 60 )) 

echo "Runtime BTC: $minutes:$seconds (mm:ss)" >> timings.txt

