import numpy as np
import nibabel as nb
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import warnings
import pickle


def normalise_like_matlab(x):
    # normalising like matlab. used for consistency with s.jbabdi's matlab code
    dim = 0
    dims = x.shape
    dimsize = dims[dim]
    dimrep = np.ones(len(dims), dtype=int)
    dimrep[dim] = dimsize
    x = x - np.tile(x.mean(axis=0), reps=dimrep)
    x = x/np.tile(x.std(axis=0, ddof=1), reps=dimrep)
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    x = x/np.sqrt(dimsize - 1)
    return x


def get_parcels(parcellation):
    # parcel "0" is not given to irrelevant vertices
    parcels = list(np.unique(parcellation))
    if 0 in parcels:
        parcels.remove(0)
    return parcels


def from_pickle(pickle_path):
    with open(pickle_path, 'rb') as pickle_in:
        data = pickle.load(pickle_in)
    return data


def read_nii(entry):
    if isinstance(entry, np.ndarray):
        # the object is already the img
        return entry
    img = nb.load(entry)
    return np.asarray(img.get_fdata())


def read_data(entry):
    if isinstance(entry, np.ndarray):
        # the object is already the img
        return entry
    if entry.endswith('.nii'):
        return read_nii(entry)
    elif entry.endswith('pickle'):
        return from_pickle(entry)


def eval_pred_success(pred_maps, real_maps, mask=None, plot=False):
    if not isinstance(mask, np.ndarray):
        print('not masking')
        mask = np.arange(pred_maps.shape[0])
    C = corrmat(pred_maps[mask,:], real_maps[mask,:])
    diag = C.diagonal()
    off_diag = C[np.triu(np.ones(C.shape)) == 0]

    return diag, off_diag, C


def corrmat(A,B):
    # create corrmat of two matrices, used for pred-VS-orig analysis
    # assumes verticesXsubject matrices, returns subjectXsubject corrmat
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    B = (B - B.mean(axis=0)) / B.std(axis=0)
    corrmat = (np.dot(B.T, A) / B.shape[0])
    return corrmat


def read_multiple_ts_data(file_paths, trim=None):
    # reads multiple time series files, normalizes, demeans and concatenates
    # trim is used if you do not wish to use all the time points in your data
    all_data = []
    for file_path in file_paths:
        rs = read_nii(file_path).T
        if isinstance(trim, np.ndarray):
            rs = rs[:,trim]
        all_data.append(detrend(StandardScaler().fit_transform(rs)))
    return np.concatenate(all_data, axis=1)


def fsl_glm(x,y):
    # adapted from s.jbabdi's matlab code. does glm the fsl way
    c = np.eye(x.shape[1])
    beta = np.matmul(np.linalg.pinv(x), y)
    cope = np.matmul(c, beta)
    r = y - np.matmul(x,beta)
    dof = r.shape[0]-np.linalg.matrix_rank(x)

    sigma_sq = np.sum(r**2, axis=0) / dof
    sigma_sq = np.expand_dims(sigma_sq, axis=0)
    bla = np.diag(np.matmul(np.matmul(c, np.linalg.inv(np.matmul(x.T,x))),c.T))
    bla = np.expand_dims(bla,axis=1)
    varcope = bla*sigma_sq
    t = np.divide(cope, np.sqrt(varcope))
    t[np.isnan(t)] = 0

    return t


def save_nii(data, tmp, fname):
    dims = data.shape
    # assume vertex dim is always bigger than time / participants dim
    # so make sure vertex dim is 1 and other dim is 0 (like nibabel needs)
    if dims[0] > dims[1]:
        data = data.T
        dims = data.shape
        print(f'transposing data before saving. current dimensions are {dims}')
    # open template and see if header dims are ok
    # if not ok, create new hdr
    tmp = nb.load(tmp)
    hdr = tmp.header

    if hdr.get_axis(0).size != dims[0]:
        s = nb.cifti2.cifti2_axes.SeriesAxis(0, 1, dims[0]) # create "time series" of the other dimention
        hdr = nb.cifti2.Cifti2Header.from_axes((s, hdr.get_axis(1)))

    to_save = nb.cifti2.cifti2.Cifti2Image(data, header=hdr)
    nb.save(to_save, fname)


def make_multi_subject_maps_obj(subjlist, data_dir, path_to_file, out_path=None, nii_tmplt=None):
    # creating matrices of concatenated
    # assumes data is arranged in this fashion:
    # data_dir includes directories with subject nums, and subsequent path_to_file is identical for all subs

    if isinstance(subjlist, str):
        with open('/Volumes/homes/Shachar/connTask_py_test/subjects.txt') as f:
            subjects = [s.strip('\n') for s in f.readlines()]
    elif isinstance(subjlist, list):
        subjects = subjlist
    else:
        warnings.warn('subjlist format not compatible')

    all_maps = np.zeros((len(subjects), 91282)) # assumes 91282 vertices
    for i, sub in enumerate(subjects):
        all_maps[i,:] = read_nii(f'{data_dir}/{sub}/{path_to_file}')

    if out_path is not None:
        if out_path.endswith('dtseries.nii'):
            save_nii(all_maps, nii_tmplt, out_path)
        elif out_path.endswith('.pickle'):
            with open(out_path, 'wb') as pickle_out:
                pickle.dump(all_maps, pickle_out)

    return all_maps


def read_all_features(subjlist, data_dir, path_to_file):
    # assumes all features files for all subs are in the same directory
    if isinstance(subjlist, str):
        with open('/Volumes/homes/Shachar/connTask_py_test/subjects.txt') as f:
            subjects = [s.strip('\n') for s in f.readlines()]
    elif isinstance(subjlist, list):
        subjects = subjlist
    else:
        warnings.warn('subjlist format not compatible')

    all_features = []
    for sub in subjects:
        all_features.append(read_data(f'{data_dir}/{sub}_{path_to_file}'))
    all_features = np.dstack(all_features)
    return all_features.transpose([1, 2, 0])  # reshape to expected dimensions

