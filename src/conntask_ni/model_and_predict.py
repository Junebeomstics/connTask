import os
import pickle
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNetCV, SGDRegressor
from sklearn.model_selection import GroupKFold

from . import utils
import nibabel as nb
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
import nilearn
import nilearn.masking

class ConnTask_sklearn:
    def __init__(self, features, target, parcellation, model_kws, normalise_features, data_type):

        if normalise_features:
            self.features = proc_features(features, data_type=data_type)  # 3D data: verticesXparticipantsXfeature_number
        else:
            self.features = utils.read_data(features)
        if self.features.shape[0] < self.features.shape[1]:
            self.features = self.features.T

        self.target = utils.read_data(target)  # 2D data: verticesXparticipants
        if self.target.shape[0] < self.target.shape[1]:
            self.target = self.target.T

        self.parcellation = utils.read_data(parcellation)
        self._parcels = utils.get_parcels(self.parcellation) # [1,2,3,...,50] (0 is removed)
        self.number_of_parcels = len(self._parcels)
        self._fit_flag = False

        if model_kws['type'] == 'glm':
            self.models = [LinearRegression() for parcel in range(self.number_of_parcels)]
        if model_kws['type'] == 'elnet':
            self.models = [ElasticNetCV(l1_ratio=model_kws['l1_ratio'], n_alphas=model_kws['n_alphas']) for parcel in range(self.number_of_parcels)]
        if model_kws['type'] == 'SGDRegressor':
            self.models = [SGDRegressor(alpha=.0001, penalty=None) for parcel in range(self.number_of_parcels)]

    def refit_model(self):
        self._fit_flag = False
        self.fit_model()

    def fit_model(self):
        if self._fit_flag:
            warnings.warn('model fitting already performed. use "self.refit_model()" if you wish to refit')
        if not len(self.features.shape) == 3:
            warnings.warn('no valid training features')
        print('fitting model per parcel')
        # fit several models, one model per parcel
        # in each model, each voxel of each participant is a "sample", with k features,
        # the target is the z-score of this voxel (of this specific participant) in the task contrast to be predicted
        for parcel in range(self.number_of_parcels):
            print(f'{parcel+1}/{self.number_of_parcels}...')
            parcel_mask = (self.parcellation == self._parcels[parcel]).flatten()
            parcel_target = self.target[parcel_mask, :].flatten()
            parcel_features = prepare_features_for_pred(self.features, parcel_mask)
            self.models[parcel].fit(parcel_features, parcel_target)
        self._fit_flag = True
        
    def partial_fit_model(self):
        if self._fit_flag:
            warnings.warn('model fitting already performed. use "self.refit_model()" if you wish to refit')
        if not len(self.features.shape) == 3:
            warnings.warn('no valid training features')
        print('fitting model per parcel')
        # fit several models, one model per parcel
        # in each model, each voxel of each participant is a "sample", with k features,
        # the target is the z-score of this voxel (of this specific participant) in the task contrast to be predicted
        for parcel in range(self.number_of_parcels):
            print(f'{parcel+1}/{self.number_of_parcels}...')
            parcel_mask = (self.parcellation == self._parcels[parcel]).flatten()
            parcel_target = self.target[parcel_mask, :].flatten() # (voxel in the parcel * samples)
            parcel_features = prepare_features_for_pred(self.features, parcel_mask) # voxels in the parcel * (samples*ICA components)
            self.models[parcel].partial_fit(parcel_features, parcel_target)
        self._fit_flag = True

    def predict(self, sub_features, normalise):
        # TODO: add an option to save image

        if normalise:
            sub_features = proc_features(sub_features)
        else:
            sub_features = utils.read_data(sub_features)
        if not self._fit_flag:
            warnings.warn('cannot predict before fitting a model')
        else:
            predicted_map = np.zeros(self.parcellation.shape).flatten()
            for parcel in range(self.number_of_parcels):
                parcel_mask = (self.parcellation == self._parcels[parcel]).flatten()
                parcel_features_ready = prepare_features_for_pred(sub_features, parcel_mask)
                predicted_map[parcel_mask] = self.models[parcel].predict(parcel_features_ready) # fit model as many as the number of parcels.
        return predicted_map

    def save_models(self, pickle_path, description):
        with open(pickle_path, 'wb') as pickle_out:
            to_save = {'models_list': self.models, 'description': description}
            pickle.dump(self.models, pickle_out)

# if model weights are already exists, use this function to get map.
def predict_map(sub_features, normalise, models, parcellation):
    if isinstance(models, str):
        with open(models, 'rb') as pickle_in:
            dic = pickle.load(pickle_in)
            models = dic['models_list']
    if normalise:
        sub_features = proc_features(sub_features)
    else:
        sub_features = utils.read_data(sub_features)
    predicted_map = np.zeros(parcellation.shape).flatten()
    parcels = utils.get_parcels(parcellation)
    for parcel in range(len(parcels)):
        parcel_mask = (parcellation == parcels[parcel]).flatten()
        parcel_features_ready = prepare_features_for_pred(sub_features, parcel_mask)
        predicted_map[parcel_mask] = models[parcel].predict(parcel_features_ready)
    return predicted_map

# custom
def proc_features(features,data_type='nifti'):
    """
    :param features: 3d or 2d matrices of Vertices X (participants) X features.
    could be a path to a dtseries.nii file
    :return: features matrix demeaned and normalised
    """
    # utils.read_data(features)
    print('normalising features...')
    if data_type == 'cifti':
        ctx = np.arange(59412) # HCP has the first 59412 voxels as cortex.
        subctx = np.setdiff1d(np.arange(91282), ctx)
        if len(features.shape) == 3:
            features[ctx, :, :] = utils.normalise_like_matlab(features[ctx, :, :])
            features[subctx, :, :] = utils.normalise_like_matlab(features[subctx, :, :])
        elif len(features.shape) == 2:
            features[ctx, :] = utils.normalise_like_matlab(features[ctx, :])
            features[subctx, :] = utils.normalise_like_matlab(features[subctx, :])   
    features = utils.normalise_like_matlab(features)
    return features


def prepare_features_for_pred(features, parcel_mask):
    """
    returns masked and demeaned features
    """
    if len(features.shape) == 3: # features contain several subjects
        parcel_features = features[parcel_mask,:,:]
        parcel_features = parcel_features - parcel_features.mean(axis=0)[np.newaxis, :, :]
        dims = parcel_features.shape
        parcel_features = np.reshape(parcel_features, [dims[0]*dims[1], dims[2]]) # (voxels * subjects, features )
    elif len(features.shape) == 2: # features contain single subject
        parcel_features = features[parcel_mask,:]
        parcel_features = parcel_features - parcel_features.mean(axis=0)

    return parcel_features


class ConnTaskCV:
    def __init__(self, features, target, parcellation, model_kws,
                 normalise_features, n_splits, groups=None,
                 save_dir=None, save_pred_maps=False, save_models=False, data_type='nifti'):
        self.data_type = data_type
        if type(features) == list:
            self.features = np.array(features)
        else:
            if normalise_features:
                self.features = proc_features(features,data_type=self.data_type)  # 3D data: verticesXparticipantsXfeature_number
            else:
                self.features = utils.read_data(features)
            if self.features.shape[0] < self.features.shape[1]:
                self.features = self.features.T

        self.target = utils.read_data(target)  # 2D data: verticesXparticipants
        if self.target.shape[0] < self.target.shape[1]:
            self.target = self.target.T

        self.parcellation = utils.read_data(parcellation)
        self._parcels = utils.get_parcels(self.parcellation)
        self.number_of_parcels = len(self._parcels)
        self._fit_flag = False

        self.model_kws = model_kws
        self.splitter = GroupKFold(n_splits=n_splits)
        if groups is not None:
            self.groups = groups
        else:
            self.groups = np.arange(self.target.shape[1]) # each subject is a group, i.e, no grouping constrains on the splitter

        self.save_dir = save_dir
        self.save_pred_maps = save_pred_maps
        self.save_models = save_models
        if isinstance(save_dir, str):
            self.set_save_dir(save_dir)

        self.pred_maps = np.zeros(self.features.shape[0:2])

    def set_save_dir(self):
        if not os.isdir(self.save_dir):
            os.mkdir(self.save_dir)
            os.mkdir(f'{self.save_dir}/cv_models')

    def predict_CV(self):
        for fold, (train_indices, test_indices) in enumerate(self.splitter.split(X=self.target.T, y=None, groups=self.groups)):
            print(f'-----fold {fold+1}-----')
            train_features, train_target = self.features[:, train_indices, :], self.target[:, train_indices] # train features : voxels, subjects, components
            model = ConnTask_sklearn(features=train_features, target=train_target, parcellation=self.parcellation,
                                     model_kws=self.model_kws, normalise_features=False, data_type=self.data_type)
            model.fit_model()
            print('predicting test set...')
            for idx in test_indices:
                self.pred_maps[:, idx] = model.predict(self.features[:, idx, :], normalise=False) # self.pred_maps : voxels * subjects
                
            if self.save_models:
                model_path = f'{self.save_dir}/cv_models/models_{fold}'
                model.save_models(model_path, description=f'cv_{fold}')
        if self.save_pred_maps:
            # add saving method later
            pass
#     def predict_CV_partial_fit(self, data_dir,path_to_file):
#         for fold, (train_indices, test_indices) in enumerate(self.splitter.split(X=self.target.T, y=None, groups=self.groups)):
#             print(f'-----fold {fold+1}-----')
            
#             chunk_num = len(train_indices) // 1000  # Compute the size of each chunk
#             chunk_rest = len(train_indices) % 1000
#             chunks = [train_indices[i:i+1000] if (i // 1000) < chunk_num else train_indices[i:i+chunk_rest] for i in range(0, len(train_indices), 1000) ]
            
#             for chunk in chunks:
#                 train_features = utils.read_all_features(subjlist=self.features[chunk].tolist(), data_dir= data_dir, path_to_file=path_to_file) # train features : voxels, subjects, components
#                 train_target = self.target[:, chunk]
            
#                 model = ConnTask_sklearn(features=train_features, target=train_target, parcellation=self.parcellation,
#                                          model_kws=self.model_kws, normalise_features=False, data_type=self.data_type)
#                 model.partial_fit_model(train_features, train_target)
#                 print(f'predicting chunk {chunk}/{chunk_num}')
#             print('predicting test set...')
#             for idx in test_indices:
#                 test_features = utils.read_all_features(subjlist=self.features[idx].tolist(), data_dir= data_dir, path_to_file=path_to_file)
#                 self.pred_maps[:, idx] = model.predict(self.features[:, idx, :], normalise=False) # self.pred_maps : voxels * subjects
                
#             if self.save_models:
#                 model_path = f'{self.save_dir}/cv_models/models_{fold}'
#                 model.save_models(model_path, description=f'cv_{fold}')
#         if self.save_pred_maps:
#             # add saving method later
#             pass

        

class ConnTaskCV_v2:
    def __init__(self, data, parcellation, model_kws,
                 normalise_features, n_splits, groups=None,
                 save_dir=None, save_pred_maps=False, save_models=False, data_type='nifti',mask_dir=None):
        
        self.data_type = data_type
        self.mask_dir = mask_dir
        self.normalise_features = normalise_features
        
        self.data = np.array(data)
        
#         if type(features) == list:
#             self.features = np.array(features)
#         else:
#             if normalise_features:
#                 self.features = proc_features(features,data_type=self.data_type)  # 3D data: verticesXparticipantsXfeature_number
#             else:
#                 self.features = utils.read_data(features)
#             if self.features.shape[0] < self.features.shape[1]:
#                 self.features = self.features.T

#         self.target = utils.read_data(target)  # 2D data: verticesXparticipants
#         if self.target.shape[0] < self.target.shape[1]:
#             self.target = self.target.T

        self.parcellation = utils.read_data(parcellation)
        self._parcels = utils.get_parcels(self.parcellation)
        self.number_of_parcels = len(self._parcels)
        self._fit_flag = False

        self.model_kws = model_kws
        self.splitter = GroupKFold(n_splits=n_splits)
        if groups is not None:
            self.groups = groups
        else:
            self.groups = np.arange(len(data)) # each subject is a group, i.e, no grouping constrains on the splitter

        self.save_dir = save_dir
        self.save_pred_maps = save_pred_maps
        self.save_models = save_models
        if isinstance(save_dir, str):
            self.set_save_dir(save_dir)
            
        nvoxels = (nb.load(self.mask_dir).get_fdata() != 0).sum()

        self.pred_maps = np.zeros((nvoxels,len(data))) # vertices * subjects
        self.target = np.zeros((nvoxels,len(data))) # vertices * subjects

    def set_save_dir(self):
        if not os.isdir(self.save_dir):
            os.mkdir(self.save_dir)
            os.mkdir(f'{self.save_dir}/cv_models')
            
    def get_params(self,batch_size=128,num_workers=128,drop_last=False):
            return {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "drop_last": drop_last,
            }

    def predict_CV_partial_fit(self):
        for fold, (train_indices, test_indices) in enumerate(self.splitter.split(X=self.data, y=None, groups=self.groups)):
            print(f'-----fold {fold+1}-----')
            train_data = self.data[train_indices] # [[rs_dir1,task_dir1],[rs_dir2,task_dir2],[rs_dir3,task_dir3]...]
            test_data = self.data[test_indices] # [[rs_dir1,task_dir1],[rs_dir2,task_dir2],[rs_dir3,task_dir3]...]
            
            train_dataset = BaseDataset(data = train_data, mask_dir = self.mask_dir)
            test_dataset = BaseDataset(data = test_data, mask_dir = self.mask_dir)
            train_loader = DataLoader(train_dataset, **self.get_params(), sampler=RandomSampler(train_dataset))
            test_loader = DataLoader(test_dataset, **self.get_params(batch_size=1), sampler=None)
            
            for i,(train_features,train_target) in enumerate(train_loader):
                #print(f'processing batch: {i}/{len(train_loader)}')
                # train_features should be (participants-X-feature_number-X-vertices)
                #print(train_features.shape)
                train_features = train_features.permute([2, 0, 1]).numpy() # (vertices -X-participants-X-feature_number)
                #print(train_features.shape)
                train_target = train_target.numpy()
                
                model = ConnTask_sklearn(features=train_features, target=train_target, parcellation=self.parcellation,
                                         model_kws=self.model_kws, normalise_features=True, data_type=self.data_type)
                model.partial_fit_model()
        
            print('predicting test set...')
            for i,(test_feature,test_target) in enumerate(test_loader):
                #print(f'processing batch: {i}/{len(test_loader)}')
                test_feature = test_feature.permute([2, 0, 1]).numpy() # (vertices -X-participants-X-feature_number)
                # test_features = utils.read_all_features(subjlist=self.features[idx].tolist(), data_dir= data_dir, path_to_file=path_to_file)
                self.pred_maps[:, test_indices[i]] = model.predict(test_feature[:,0,:], normalise=True) # self.pred_maps : voxels * subjects
                self.target[:, test_indices[i]] = test_target.numpy()[0,:]
            if self.save_models:
                model_path = f'{self.save_dir}/cv_models/models_{fold}'
                model.save_models(model_path, description=f'cv_{fold}')
        if self.save_pred_maps:
            # add saving method later
            pass



class BaseDataset(Dataset):
    def __init__(self,data,mask_dir):
        super().__init__()
        self.data = data
        self.mask_dir = mask_dir

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        features_dir, task_dir =self.data[index]
        features = utils.read_data(features_dir)
        
        #assume that mask_file is resized 3D nifti object
        task_map = nb.load(task_dir)
        
        MNI152_mask = nb.load(self.mask_dir)
        masked_task_map = nilearn.masking.apply_mask(task_map,mask_img=MNI152_mask)
        
        return features,masked_task_map

        