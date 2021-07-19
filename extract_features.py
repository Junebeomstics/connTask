import numpy as np
import utils
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import warnings


def parse_dr_args(data, components):
    if isinstance(data, list):
        data = utils.read_multiple_ts_data(data)
    elif isinstance(data, str):
        rs = utils.read_nii(data).T
        data = detrend(StandardScaler().fit_transform(rs))
    elif isinstance(data, np.ndarray):
        if data.shape[0] < data.shape[1]:
            data = data.T
    else:
        warnings.warn('data is supplied in unknown format')

    if isinstance(components, str):
        components = utils.read_nii(components).T
    elif isinstance(components, np.ndarray):
        if components.shape[0] < components.shape[1]:
            components = components.T

    return data, components


def dual_regression(data, components):
    data, components = parse_dr_args(data, components)
    # step 1
    pinv_comp = np.linalg.pinv(components)
    ts = np.matmul(pinv_comp, data)

    # step 2
    sub_comps = utils.fsl_glm(ts.T, data.T)
    return sub_comps.T

# TODO: write finalfeatureextraction function; wrapper class to run the process

