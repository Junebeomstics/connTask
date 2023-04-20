#Dual regression and connectivity map extraction for rs-fMRI.

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
parser.add_argument('--rs_output_file', default='features_43_comps.dtseries.nii', type=str)

args = parser.parse_args()

cifti = nb.load(args.groupICA_file) # i keep t
ica = np.asarray(cifti.get_fdata())

# set an outpurt directory for the features
outdir = args.outdir
# set origin directory for raw data
rs_data_dir = args.rs_data_dir

# iterate through all participants and perform feature extraction
subjects = [ subj for subj in os.listdir(rs_data_dir) if len(os.listdir(os.path.join(rs_data_dir,subj))) > 0 ] #'/Volumes/homes/Shachar/connTask_py_test/subjects.txt'
# with open(subjlist) as f:
#     subjects = [s.strip('\n') for s in f.readlines()]

for i, sub in enumerate(sorted(subjects)):
    rs_dir = f'{rs_data_dir}/{sub}'
    print(f'subject {i+1}/{len(subjects)}: {sub}')
    
    if not os.path.exists(f'{outdir}/{sub}_{args.rs_output_file}'): 
        # read data from all 4 runs of the hcp-data.
        # read_multiple_ts_data() normalizes, demeans, detrends, and concatenates
        # the data from the different runs
        rs_paths = [f'{rs_dir}/{scan}' for scan in os.listdir(rs_dir)]
        data = utils.read_multiple_ts_data(rs_paths)

        # perform the two steps of feature-extraction
        print('extracting features')
        dr_comps = extract_features.dual_regression(data, ica)
        print(dr_comps.shape)
        print(data.shape)
        features = extract_features.weighted_seed2voxel(dr_comps, data)

        # save to outdir
        print('saving features')
        to_save = nb.cifti2.cifti2.Cifti2Image(features.T, header=cifti.header)
        nb.save(to_save, f'{outdir}/{sub}_{args.rs_output_file}')
    
