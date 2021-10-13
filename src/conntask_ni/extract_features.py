import numpy as np
from conntask_ni import utils
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import warnings


def parse_dr_args(data, components):
    # read data and make sure its in the right dimensions for further processing
    if isinstance(data, list):
        data = utils.read_multiple_ts_data(data)
    elif isinstance(data, str):
        rs = utils.read_data(data).T
        data = detrend(StandardScaler().fit_transform(rs))
    elif isinstance(data, np.ndarray):
        if data.shape[0] < data.shape[1]:
            data = data.T
    else:
        warnings.warn('data is supplied in unknown format')

    if isinstance(components, str):
        components = utils.read_data(components).T
    elif isinstance(components, np.ndarray):
        pass
    else:
        print('ica components should be a path to an nii file or an nd.array')
        raise TypeError
    if components.shape[0] < components.shape[1]:
        components = components.T

    return data, components


def dual_regression(data, group_components):
    # perform dual regression to yield subject-specific spatial-components
    data, group_components = parse_dr_args(data, group_components)
    # step 1: get components subject-specific time series for each components
    pinv_comp = np.linalg.pinv(group_components)
    ts = np.matmul(pinv_comp, data)
    # step 2: find the subject-specific spatial pattern for each component
    sub_comps = utils.fsl_glm(ts.T, data.T)
    return sub_comps.T


def weighted_seed2voxel(seeds, data):
    # performs a weighted_seed2voxel analysis for each component
    # to yield the connectivity maps used in the connTask prediction pipeline
    pinv_seeds = np.linalg.pinv(seeds)
    ts = np.matmul(pinv_seeds, data)
    ts_norm = utils.normalise_like_matlab(ts.T).T
    data_norm = utils.normalise_like_matlab(data.T).T
    features = np.matmul(ts_norm, data_norm.T).T
    return features

