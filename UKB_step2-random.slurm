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

python UKB_step2-random.py --num_samples 3000 --task_output_file result/UKB/emotion_1_target_3000samples_Sch_100P_7N.pickle --maskdir MNI_152_mask.nii.gz --parcel_dir masked_ROI_Sch_100P_7N.npy --output_name output-random --fig_name UKB_d21-random.png
