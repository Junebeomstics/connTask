# origin (45 features)
from src.conntask_ni import utils, extract_features, model_and_predict
import nibabel as nb
import numpy as np
import scipy
import seaborn as sns
import os
import pickle
import csv

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--ICA_components', default=21, type=int,choices=[21,55])
parser.add_argument('--num_samples', default=-1, type=int, help='')
parser.add_argument('--cope', default=1, type=int)
parser.add_argument('--model', default='glm', type=str)

parser.add_argument('--rs_data_dir', default='/global/cfs/cdirs/m4244/registration/20227_1_MNI', type=str)
parser.add_argument('--task_data_dir', default='/global/cfs/cdirs/m4244/registration/20249_unzip_1_beta_MNI', type=str)
parser.add_argument('--maskfile', default='MNI_152_mask.nii.gz', type=str)
parser.add_argument('--parcel_file', default='masked_ROI_Sch_100P_7N.npy', type=str)


parser.add_argument('--output_base', default='output/UKB', type=str)


args = parser.parse_args()

args.rs_data_dir = os.path.join(args.output_base,f'features-d{args.ICA_components}')
args.rs_output_file = os.path.join(f'features_{args.ICA_components}_comps.npy')
args.groupICA_file = f'UKB_groupica_{args.ICA_components}_comp_masked.npy'
args.fig_name = os.path.join(args.output_base,'fig',f'UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}.png')
args.result_file = os.path.join(args.output_base, 'result',f'output_UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}.csv')
args.pred_maps = os.path.join(args.output_base, 'result',f'output_UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}_pred_maps.pickle')
args.matrix_dir = os.path.join(args.output_base, 'result',f'output_UKB_cope{args.cope}_d{args.ICA_components}_nsamples_{args.num_samples}_{args.model}.npy')
args.task_output_file = os.path.join(args.output_base,'task',f'task_emotion_cope{args.cope}_nsamples_{args.num_samples}.pickle')
args.parcel_dir = os.path.join(args.output_base,'ROI',args.parcel_file)
args.maskdir = os.path.join(args.output_base,'mask',args.maskfile)


# set origin directory for raw data
rs_data_dir = args.rs_data_dir

task_data_dir = args.task_data_dir
all_betas = sorted(os.listdir(task_data_dir))
rsfMRI = [ subj[:17] for subj in os.listdir(rs_data_dir) ] 

subjlist_20249 = [subj for subj in all_betas if (os.path.exists(os.path.join(task_data_dir,subj,f'zstat{args.cope}_MNI_space.nii.gz')) and (subj.replace('20249','20227') in rsfMRI))] # filter out subjects that has both task fMRI and rs-fMRI

subjlist_20227 = [subj.replace('20249','20227') for subj in subjlist_20249]

subjlist_20249 = subjlist_20249[:args.num_samples]
subjlist_20227 = subjlist_20227[:args.num_samples]


path_to_file = f'zstat{args.cope}_MNI_space.nii.gz'

# target = utils.make_multi_subject_maps_obj(subjlist=subjlist, data_dir=task_data_dir,path_to_file=path_to_file, out_path=args.task_output_file)


if not os.path.exists(args.task_output_file):
    print('processing task fMRI contrasts')
    target = utils.make_multi_subject_maps_obj_for_UKB(subjlist=subjlist_20249, data_dir=task_data_dir,mask_dir=args.maskdir,path_to_file=path_to_file, out_path=args.task_output_file)
    print('processing task fMRI contrasts finished')
else:
    print('loading task fMRI contrasts')
    with open(args.task_output_file, 'rb') as pickle_in:
        target = pickle.load(pickle_in)
print('loading and processing resting-state fMRI features')
# read in all subject's rs-fMRI features to one 3d-matrix (vertices-X-participants-X-feature_number)
# this could take a while, as it's a large amount of data...
features = utils.read_all_features(subjlist=subjlist_20227, data_dir= rs_data_dir, path_to_file=args.rs_output_file)
print('finished loading features')

# set needed parameters
parcellation = utils.read_data(args.parcel_dir) # should be (nvoxels)
model_kws = {'type': args.model}
n_splits = 5

# form the connTaskCV object
cv_predict = model_and_predict.ConnTaskCV(features=features, target=target, parcellation=parcellation,model_kws=model_kws, normalise_features=True, n_splits=n_splits, data_type='nifti') # you can specify 'cifti' for HCP cifti dataset
print('model and predict using CV')
cv_predict.predict_CV()


# as we only performed prediction in areas included in the parcellation
# we shall only examine prediction sucess in these areas
mask = (parcellation>0).flatten()


        
with open(args.pred_maps, 'wb') as pickle_out:
    pickle.dump(cv_predict.pred_maps, pickle_out)

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
data = [args.model, args.ICA_components,args.cope, float(diag.mean()),float(diag.std()),float(np.median(diag)) ,float(diag.mean() - off_diag.mean()), KS_D,KS_pvalue]

# Save the data to a CSV file
with open(args.result_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model','ICA_components','cope','Diagonal Mean','Diagonal Std','Diagonal Median', 'Diagonality Index','KS_D','KS_pvalue'])
    writer.writerow(data)




