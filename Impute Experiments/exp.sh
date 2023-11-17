#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 110:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot2-impute
#SBATCH -p defq
#SBATCH --exclusive
#SBATCH --exclude=esplhpc-cp040
#SBATCH --mail-type = FAIL, BEGIN, END
#SBATCH --mail-user = Gabriel.Ketron@cshs.org
#SBATCH --mail-user=gketron@uci.edu
#SBATCH -o ./logs/output.%j_%a.out # STDOUT
#SBATCH --array=1-5

source /common/ketrong/minconda3/etc/profile.d/conda.sh

'''
conda deactivate myenv12
conda remove --name myenv12 --all
conda create --name myenv12 -c conda-forge python=3.10
'''
conda activate tpot2env

pip install -r requirements.txt

echo RunStart

srun -u /home/ketrong/miniconda3/envs/tpot2env/bin/python main.py \
--n_jobs 48 \
--savepath results/ \
--num_runs 5 \