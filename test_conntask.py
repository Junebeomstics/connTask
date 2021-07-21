import model_and_predict
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.io
import utils

feature_dir = '/Volumes/home/Movie_vs_Rest/Feature_extraction/Movie_all/FF_cleaned'
subject_list = '/Volumes/home/Movie_vs_Rest/all_subjects_7T_3T.txt'
with open(subject_list, 'r') as f:
    subjects = [s.strip('\n') for s in f.readlines()]

all_features = []
for sub in subjects:
    all_features.append(model_and_predict.read_nii(f'{feature_dir}/{sub}_RFMRI_nosmoothing.dtseries.nii'))

all_features = np.dstack(all_features)
all_features = model_and_predict.proc_features(all_features.transpose([1, 2, 0])) # reshape and normalise
all_target = utils.read_nii('/Volumes/home/Movie_vs_Rest/orig_data/WM_09_s4_all_data_z.dtseries.nii')
all_target = all_target.transpose()
parcellation_ica = model_and_predict.read_nii('/Volumes/HCP/Parcellation/ica_parcellation_50.dtseries.nii')
parcellation_schafer = model_and_predict.read_nii('/Volumes/HCP/Parcellation/schafer/Schaefer2018_100Parcels_7Networks_order.dtseries.nii')

train_features = all_features[:, np.arange(1,159), : ]
train_target = all_target[:, np.arange(1,159)]

# make model
glm_kws = {'type': 'glm'}
elnet_kws = {'type': 'elnet', 'l1_ratio': [0.01, .1, .2, .3, .4, .5, .7, .9, .95, .99, 1], 'n_alphas':100}
model = model_and_predict.ConnTask_sklearn(train_features=train_features, parcellation=parcellation_schafer,
                                           train_target=train_target, model_kws=glm_kws, normalise_features=False)
model.fit_model()
sub_pred_img = model.predict(all_features[:, 0 ,:], normalise=False)

# predict and assess
sub_pred_img = model_and_predict.predict_map(all_features[:, 0 ,:], normalise=False, models='test_files/models.pickle' , parcellation=parcellation_schafer)

ctx = np.arange(59412)
subctx = np.setdiff1d(np.arange(91282), ctx)
mask = (parcellation_ica>0).flatten()
mask[subctx] = 0

pearsonr(sub_pred_img[mask], all_target[mask,0])

# test cv
cv_predict = model_and_predict.ConnTaskCV(features=all_features, target=all_target, parcellation=parcellation_ica,
                                          model_kws=glm_kws, normalise_features=False, n_splits=5)
cv_predict.predict_CV()
diag, off_diag, CM = model_and_predict.eval_pred_success(cv_predict.pred_maps, cv_predict.target, mask)
diag.mean()
off_diag.mean()


# all_features_normed = all_features_reshaped
# all_features_normed[ctx,:,:] = model_and_predict.normalise_like_matlab(all_features_normed[ctx,:,:])
# all_features_normed[subctx,:,:] = model_and_predict.normalise_like_matlab(all_features_normed[subctx,:,:])



