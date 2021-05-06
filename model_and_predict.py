import numpy as np
import pandas as pd
import nibabel as nb
from sklearn.linear_model import LinearRegression
from typing import Optional
import warnings

def get_parcels(parcellation):
    parcels = np.unique(parcellation)
    if 0 in parcels:
        return parcels.remove(0)
    else:
        return parcels


class connTaskmodel_glm:
    def __init__(self, train_features, train_target, parcellation, description=None):
        # TODO: add option to give paths (i.e, strings) instead of images, and read the files using the module
        self.train_features = train_features  # 3D data: participantsXverticesXfeature_number
        self.train_target = train_target  # 2D data: participantsXvertices
        self.parcellation = parcellation
        self._parcels = get_parcels(self.parcellation)
        self.number_of_parcels = len(self._parcels)
        self.models = [LinearRegression() for parcel in range(self.number_of_parcels)]
        self.description = description  # string or dict with some information regarding the instance
        self._fit_flag = False

    def _prepare_features_for_pred(self, parcel_features):
        #  do stuff
        #  return a (normalized+demeaned) matrix of [(vertices*participants)Xfeatures]
        pass

    def fit_model(self):
        # fit several models, one model per parcel
        # in each model, each voxel of each participant is a "sample", with k features,
        # the target is the z-score of this voxel (of this specific participant) in the task contrast to be predicted
        for parcel in range(self.number_of_parcels):
            parcel_mask = self.parcellation == self._parcels[parcel]
            parcel_target = self.train_target[:, parcel_mask]
            parcel_target = parcel_target.flatten()
            parcel_features = self.train_features[:, parcel_mask, :]
            parcel_features_ready = self._prepare_features_for_pred(parcel_features)
            self.models[parcel].fit(parcel_features_ready, parcel_target)
        self._fit_flag = True

    def predict(self, sub_features, out_path):
        if not self._fit_flag:
            warnings.warn('cannot predict before fitting a model')
        else:
            predicted_map = np.zeros(self.parcellation.shape)
            for parcel in range(self.number_of_parcels):
                parcel_mask = self.parcellation == self._parcels[parcel]
                parcel_features = sub_features[parcel_mask]
                parcel_features_ready = self._prepare_features_for_pred(parcel_features)
                predicted_map[parcel_mask] = self.models[parcel].predict(parcel_features_ready)
            if out_path:
                # save map
        return predicted_map






