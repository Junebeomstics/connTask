# origin (45 features)
from conntask_ni import utils, extract_features, model_and_predict
import nibabel as nb
import numpy as np
import seaborn as sns
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--groupICA_file', default='src/conntask_ni/files/ica_both_hemis_45_comp_cleaned.dtseries.nii', type=str)
parser.add_argument('--outdir', default='connTask_py_test/features-d43', type=str)
parser.add_argument('--rs_data_dir', default='hcp1200_ICA/data', type=str)
parser.add_argument('--task_data_dir', default='hcp1200_betamap/data', type=str)
parser.add_argument('--cope', default='cope1', type=str)
parser.add_argument('--task_output_file', default='connTask_py_test/emotion_01_target.pickle', type=str)
parser.add_argument('--rs_output_file', default='features_43_comps.dtseries.nii', type=str)
parser.add_argument('--parcel_dir', default='src/conntask_ni/files/Schaefer2018_100Parcels_7Networks_order.dtseries.nii', type=str)
parser.add_argument('--fig_name', default='d43.png', type=str)

args = parser.parse_args()

# set an outpurt directory for the features
outdir = args.outdir
# set origin directory for raw data
rs_data_dir = args.rs_data_dir

task_data_dir = args.task_data_dir
all_betas = os.listdir(task_data_dir)
rsfMRI = [ subj for subj in os.listdir(rs_data_dir) if len(os.listdir(os.path.join(rs_data_dir,subj))) > 0 ] 

subjlist = [subj for subj in all_betas if (os.path.exists(os.path.join(task_data_dir,subj,args.cope)) and len(os.listdir(os.path.join(task_data_dir,subj,args.cope))) > 0 ) and (subj in rsfMRI)]
path_to_file = args.cope+'/zstat1.dtseries.nii'

target = utils.make_multi_subject_maps_obj(subjlist=subjlist, data_dir=task_data_dir,path_to_file=path_to_file, out_path=args.task_output_file)

# read in all subject's features to one 3d-matrix (vertices-X-participants-X-feature_number)
# this could take a while, as it's a large amount of data...
features = utils.read_all_features(subjlist=subjlist, data_dir= outdir, path_to_file=args.rs_output_file)


# set needed parameters
parcellation = utils.read_data(args.parcel_dir)
model_kws = {'type': 'glm'}
n_splits = 100

# form the connTaskCV object
cv_predict = model_and_predict.ConnTaskCV(features=features, target=target, parcellation=parcellation,model_kws=model_kws, normalise_features=True, n_splits=n_splits)
# model and predict using CV
cv_predict.predict_CV()


# as we only performed prediction in areas included in the parcellation
# we shall only examine prediction sucess in these areas
mask = (parcellation>0).flatten()

diag, off_diag, CM = utils.eval_pred_success(cv_predict.pred_maps, cv_predict.target, mask) # cv_predict.pred_maps : voxels , subjects, # cv_predict.target : voxels , subjects

print(f'diagonal mean: {diag.mean()}')
print(f'diagonlity index: {diag.mean() - off_diag.mean()}')
print(f'diagonal mean: {diag.mean()}')
print(f'diagonlity index: {diag.mean() - off_diag.mean()}')


heatmap = sns.heatmap(CM)
fig = heatmap.get_figure()
fig.savefig(args.fig_name) 