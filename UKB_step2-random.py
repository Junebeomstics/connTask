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
parser.add_argument('--groupICA_file', default='UKB_groupica_21_comp_masked.npy', type=str)
parser.add_argument('--outdir', default='result/UKB/features-d21', type=str)
parser.add_argument('--num_samples', default=-1, type=int, help='')
parser.add_argument('--rs_data_dir', default='/global/cfs/cdirs/m4244/registration/20227_1_MNI', type=str)
parser.add_argument('--task_data_dir', default='/global/cfs/cdirs/m4244/registration/20249_unzip_1_beta_MNI', type=str)
parser.add_argument('--maskdir', default='/global/cfs/cdirs/m4244/registration/icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.nii', type=str)
parser.add_argument('--cope', default=1, type=int)
parser.add_argument('--task_output_file', default='result/UKB/emotion_1_target_3000samples.pickle', type=str)
parser.add_argument('--rs_output_file', default='features_21_comps.npy', type=str)
parser.add_argument('--parcel_dir', default='masked_AAL_ROI.npy', type=str)
parser.add_argument('--fig_name', default='UKB_d21.png', type=str)
parser.add_argument('--output_name', default='output_UKB_d21', type=str)

args = parser.parse_args()

# set an outpurt directory for the features
outdir = args.outdir
# set origin directory for raw data
rs_data_dir = args.rs_data_dir

task_data_dir = args.task_data_dir
all_betas = sorted(os.listdir(task_data_dir))
rsfMRI = [ subj[:17] for subj in os.listdir(outdir) ] 

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
print('making random fMRI features')
# read in all subject's rs-fMRI features to one 3d-matrix (vertices-X-participants-X-feature_number)
# this could take a while, as it's a large amount of data...
# features = utils.read_all_features(subjlist=subjlist_20227, data_dir= outdir, path_to_file=args.rs_output_file)
features = np.random.randn(target.shape[1],target.shape[0],21) # (vertices-X-participants-X-feature_number)
print(features.shape)


# set needed parameters
parcellation = utils.read_data(args.parcel_dir) # should be (1,nvoxels)
model_kws = {'type': 'glm'}
n_splits = 5

# form the connTaskCV object
cv_predict = model_and_predict.ConnTaskCV(features=features, target=target, parcellation=parcellation,model_kws=model_kws, normalise_features=True, n_splits=n_splits)
print('model and predict using CV')
cv_predict.predict_CV()


# as we only performed prediction in areas included in the parcellation
# we shall only examine prediction sucess in these areas
mask = (parcellation>0).flatten()


diag, off_diag, CM = utils.eval_pred_success(cv_predict.pred_maps, cv_predict.target, mask) # cv_predict.pred_maps : voxels , subjects, # cv_predict.target : voxels , subjects

print(f'diagonal mean: {diag.mean()}')
print(f'diagonlity index: {diag.mean() - off_diag.mean()}')

heatmap = sns.heatmap(CM)
fig = heatmap.get_figure()
fig.savefig(args.fig_name) 
np.save('args.output_name'+'_corrmap.npy',CM)

# Combine the diagonal mean and index into a list
data = [diag.mean(), diag.mean() - off_diag.mean()]

# Save the data to a CSV file
with open(args.output_name+'.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Diagonal Mean', 'Diagonality Index'])
    writer.writerows(data)



