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
pip install conntask_ni

python hcp_step2.py --groupICA_file hcp1200_ICA/groupICA/groupICA_3T_HCP1200_MSMALL_d50_melodic_IC.dscalar.nii --outdir connTask_py_test/features-d50 --cope cope2 --task_output_file connTask_py_test/emotion_2_target.pickle --rs_output_file features_50_comps.dtseries.nii --fig_name d50-emotion-cope2.png
