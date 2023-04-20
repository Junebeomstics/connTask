#!/bin/bash
#SBATCH -A m4244
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --exclusive
#SBATCH --output=R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr

source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN
#pip install conntask_ni

python UKB_step1.py --groupICA_file output/UKB/groupICA/UKB_groupica_55_comp_masked.npy --outdir result/UKB/features-d55 --rs_output_file features_55_comps --start_idx 0
