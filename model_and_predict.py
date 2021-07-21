import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNetCV
import warnings
import pickle
from sklearn.model_selection import GroupKFold
import os
import utils


class ConnTask_sklearn:
    def __init__(self, features, target, parcellation, model_kws, normalise_features):

        if normalise_features:
            self.features = proc_features(features)  # 3D data: verticesXparticipantsXfeature_number
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

        if model_kws['type'] == 'glm':
            self.models = [LinearRegression() for parcel in range(self.number_of_parcels)]
        if model_kws['type'] == 'elnet':
            self.models = [ElasticNetCV(l1_ratio=model_kws['l1_ratio'], n_alphas=model_kws['n_alphas']) for parcel in range(self.number_of_parcels)]

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
                predicted_map[parcel_mask] = self.models[parcel].predict(parcel_features_ready)
        return predicted_map

    def save_models(self, pickle_path, description):
        with open(pickle_path, 'wb') as pickle_out:
            to_save = {'models_list': self.models, 'description': description}
            pickle.dump(self.models, pickle_out)


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


def proc_features(features):
    """
    :param features: 3d or 2d matrices of Vertices X (participants) X features.
    could be a path to a dtseries.nii file
    :return: features matrix demeaned and normalised
    """
    utils.read_data(features)
    print('normalising features...')
    ctx = np.arange(59412)
    subctx = np.setdiff1d(np.arange(91282), ctx)
    if len(features.shape) == 3:
        features[ctx, :, :] = utils.normalise_like_matlab(features[ctx, :, :])
        features[subctx, :, :] = utils.normalise_like_matlab(features[subctx, :, :])
        features = utils.normalise_like_matlab(features)
    elif len(features.shape) == 2:
        features[ctx, :] = utils.normalise_like_matlab(features[ctx, :])
        features[subctx, :] = utils.normalise_like_matlab(features[subctx, :])
        features = utils.normalise_like_matlab(features)
    return features


def prepare_features_for_pred(features, parcel_mask):
    """
    returns masked and demeaned features
    """
    if len(features.shape) == 3:
        parcel_features = features[parcel_mask,:,:]
        parcel_features = parcel_features - parcel_features.mean(axis=0)[np.newaxis, :, :]
        dims = parcel_features.shape
        parcel_features = np.reshape(parcel_features, [dims[0]*dims[1], dims[2]])
    elif len(features.shape) == 2:
        parcel_features = features[parcel_mask,:]
        parcel_features = parcel_features - parcel_features.mean(axis=0)

    return parcel_features


class ConnTaskCV:
    def __init__(self, features, target, parcellation, model_kws,
                 normalise_features, n_splits, groups=None,
                 save_dir=None, save_pred_maps=False, save_models=False):
        if normalise_features:
            self.features = proc_features(features)  # 3D data: verticesXparticipantsXfeature_number
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
            train_features, train_target = self.features[:, train_indices, :], self.target[:, train_indices]
            model = ConnTask_sklearn(features=train_features, target=train_target, parcellation=self.parcellation,
                                     model_kws=self.model_kws, normalise_features=False)
            model.fit_model()
            print('predicting test set...')
            for idx in test_indices:
                self.pred_maps[:, idx] = model.predict(self.features[:, idx, :], normalise=False)

            if self.save_models:
                model_path = f'{self.save_dir}/cv_models/models_{fold}'
                model.save_models(model_path, description=f'cv_{fold}')
        if self.save_pred_maps:
            # add saving method later
            pass




