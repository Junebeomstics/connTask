# origin (45 features)
from src.conntask_ni import utils, extract_features, model_and_predict
import nibabel as nb
import numpy as np
import seaborn as sns
import os
import pickle
import csv

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--ICA_components', default=21, type=int,choices=[21,55])
parser.add_argument('--num_samples', default=-1, type=int, help='')
parser.add_argument('--cope', default=1, type=int)
parser.add_argument('--model', default='SGDRegressor', type=str)

#parser.add_argument('--groupICA_file', default='UKB_groupica_21_comp_masked.npy', type=str)
parser.add_argument('--task_data_dir', default='/global/cfs/cdirs/m4244/registration/20249_unzip_1_beta_MNI', type=str)
parser.add_argument('--maskfile', default='MNI_152_mask.nii.gz', type=str)
parser.add_argument('--parcel_file', default='masked_ROI_Sch_100P_7N.npy', type=str)

parser.add_argument('--n_splits', default=5, type=int)

parser.add_argument('--output_base', default='output/UKB', type=str)
args = parser.parse_args()

args.rs_data_dir = os.path.join(args.output_base,f'features-d{args.ICA_components}')
args.rs_output_file = os.path.join(f'features_{args.ICA_components}_comps.npy')
args.groupICA_file = f'UKB_groupica_{args.ICA_components}_comp_masked.npy'
args.fig_name = os.path.join(args.output_base,'fig',f'UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}_partial.png')
args.result_file = os.path.join(args.output_base, 'result',f'output_UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}_partial.csv')
args.matrix_dir = os.path.join(args.output_base, 'result',f'output_UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}_partial.npy')
args.task_output_file = os.path.join(args.output_base,'task',f'task_emotion_cope{args.cope}_nsamples_{args.num_samples}.pickle')
args.parcel_dir = os.path.join(args.output_base,'ROI',args.parcel_file)
args.maskdir = os.path.join(args.output_base,'mask',args.maskfile)


# set an outpurt directory for the features
rs_data_dir = args.rs_data_dir

task_data_dir = args.task_data_dir
all_betas = sorted(os.listdir(task_data_dir))
rsfMRI = [ subj[:17] for subj in os.listdir(rs_data_dir)]  #1000246_20227_2_0

subjlist_20249 = [subj for subj in all_betas if (os.path.exists(os.path.join(task_data_dir,subj,f'zstat{args.cope}_MNI_space.nii.gz')) and (subj.replace('20249','20227') in rsfMRI))] # filter out subjects that has both task fMRI and rs-fMRI

subjlist_20227 = [subj.replace('20249','20227') for subj in subjlist_20249]

subjlist = [(os.path.join(args.rs_data_dir,f"{subj.replace('20249','20227')}_{args.rs_output_file}"),f'{task_data_dir}/{subj}/zstat{args.cope}_MNI_space.nii.gz') for subj in subjlist_20249] 

subjlist = subjlist[:args.num_samples]


# set needed parameters
parcellation = utils.read_data(args.parcel_dir) # should be (1,nvoxels)
model_kws = {'type': args.model}
# no partial fit
# model_kws = {'type': 'glm'}

# form the connTaskCV object
cv_predict = model_and_predict.ConnTaskCV_v2(data=subjlist, parcellation=parcellation, model_kws=model_kws, normalise_features=True, n_splits=args.n_splits, data_type='nifti',mask_dir=args.maskdir) # you can specify 'cifti' for HCP cifti dataset
print('model and predict using CV')
cv_predict.predict_CV_partial_fit()


# as we only performed prediction in areas included in the parcellation
# we shall only examine prediction sucess in these areas
mask = (parcellation>0).flatten()

diag, off_diag, CM = utils.eval_pred_success(cv_predict.pred_maps, cv_predict.target, mask) # cv_predict.pred_maps : voxels , subjects, # cv_predict.target : voxels , subjects

print(f'diagonal mean: {diag.mean()}')
print(f'diagonal std: {diag.std()}')
print(f'diagonal median: {np.median(diag)}')
print(f'diagonlity index: {diag.mean() - off_diag.mean()}')

heatmap = sns.heatmap(CM,cmap='vlag')
fig = heatmap.get_figure()
fig.savefig(args.fig_name) 
np.save(args.matrix_dir,CM)

KS_D=scipy.stats.kstest(diag,off_diag).statistic
KS_pvalue=scipy.stats.kstest(diag,off_diag).pvalue

# Combine the diagonal mean and index into a list
data = [args.model, args.ICA_components,args.cope, float(diag.mean()),float(diag.std()),float(np.median(diag)) ,float(diag.mean() - off_diag.mean())]

# Save the data to a CSV file
with open(args.result_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model','ICA_components','cope','Diagonal Mean','Diagonal Std','Diagonal Median', 'Diagonality Index','KS_D','KS_pvalue'])
    writer.writerow(data)



