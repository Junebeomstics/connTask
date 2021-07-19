import numpy as np
import nibabel as nb
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend


def normalise_like_matlab(x):
    """
    normalisation matlab style
    """
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
    parcels = list(np.unique(parcellation))
    if 0 in parcels:
        parcels.remove(0)
    return parcels


def read_nii(entry, img_type=None):
    if isinstance(entry, np.ndarray):
        # the object is already the img
        return entry
    if img_type:
        print(f'loading {img_type}...')
    img = nb.load(entry)
    return np.asarray(img.get_fdata())


def eval_pred_success(pred_maps, real_maps, mask=None, plot=False):
    if not isinstance(mask, np.ndarray):
        print('not masking')
        mask = np.arange(pred_maps.shape[0])
    C = corrmat(pred_maps[mask,:], real_maps[mask,:])
    diag = C.diagonal()
    off_diag = C[np.triu(np.ones(C.shape)) == 0]

    return diag, off_diag, C


def corrmat(A,B):
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
    # adapted from s. jbabdi
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
