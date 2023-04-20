#Dual regression and connectivity map extraction for rs-fMRI.

from src.conntask_ni import utils, extract_features, model_and_predict
import nibabel as nb
import torchio as tio
import numpy as np
import seaborn as sns
import os
import time
import nilearn
import nilearn.image 
import nilearn.masking

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--groupICA_file', default='output/UKB/groupICA/UKB_groupica_21_comp_masked.npy', type=str, help='assume that the group ICA file is registered to MNI152, masked, and flattened. (dim: # component * # voxels except backgrounds), made in UKB_preprocess_groupICA.ipynb')
parser.add_argument('--outdir', default='output/UKB/features-d21', type=str)
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--maskdir', default='output/UKB/mask/MNI_152_mask.nii.gz', type=str,help='made in UKB_preprocess_groupICA.ipynb')
parser.add_argument('--rs_data_dir', default='/global/cfs/cdirs/m4244/registration/20227_1_MNI', type=str)
parser.add_argument('--rs_output_file', default='features_21_comps', type=str)

args = parser.parse_args()

# cifti = nb.load(args.groupICA_file) # i keep t
# ica = np.asarray(cifti.get_fdata())

group_ica = np.load(args.groupICA_file) # shape: (21,237969)

# set an outpurt directory for the features
outdir = args.outdir
# set origin directory for raw data
rs_data_dir = args.rs_data_dir
MNI152_mask = nb.load(args.maskdir)

# target_affine = nifti_orig.affine
# target_shape = nifti_orig.shape[:3]
# MNI152_mask = nilearn.image.resample_img(mask_file_nifti,target_affine=target_affine,target_shape=target_shape,interpolation='nearest')



# iterate through all participants and perform feature extraction
subjects = [ subj for subj in os.listdir(rs_data_dir)] #'/Volumes/homes/Shachar/connTask_py_test/subjects.txt'
# with open(subjlist) as f:
#     subjects = [s.strip('\n') for s in f.readlines()]

for i, sub in enumerate(sorted(subjects)[args.start_idx:]):
    rs_file = f'{rs_data_dir}/{sub}'
    print(f'subject {i+1}/{len(subjects)}: {sub[:17]}')
    
    start=time.time()
    try:
        if not os.path.exists(f'{outdir}/{sub[:17]}_{args.rs_output_file}.npy'): 
            # read data from all 1 runs of the UKB-data.
            # read_multiple_ts_data() normalizes, demeans, detrends, and concatenates
            # only one run


            rs_image = nb.load(rs_file)
            masked_pixels = nilearn.masking.apply_mask(rs_image, mask_img=MNI152_mask)

            # masked_pixels: np.ndarray((490,237969))
            # input to the 'utils.read_multiple_ts_data' should be ntimepoints * nvoxels.

            rs_paths = [masked_pixels]

            data = utils.read_multiple_ts_data(rs_paths)

            # perform the two steps of feature-extraction
            print('extracting features')

            dr_comps = extract_features.dual_regression(data, group_ica)
            # print(dr_comps.shape)
            # print(data.shape)
            features = extract_features.weighted_seed2voxel(dr_comps, data)

            # save to outdir
            print('saving features')
            to_save = features.T
            #to_save = nb.cifti2.cifti2.Cifti2Image(features.T, header=cifti.header)
            np.save(f'{outdir}/{sub[:17]}_{args.rs_output_file}', to_save)
    except:
        print(f'have problems in file {sub}')
        with open('corrupted_files.txt','a') as f:
            f.write(sub+'\n')
    end=time.time()
    print(f'time taken to process {sub[:17]}: {end-start}')
    
