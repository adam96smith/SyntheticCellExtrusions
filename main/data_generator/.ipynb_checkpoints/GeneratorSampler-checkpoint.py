'''

First Script to Run to Create Samplers for Synthetic Images.

Identify all labelled images (volume or slice labels) and sample the fluorescence based on the distance to membrane.

Output: 
 - Fluorescent samplers that will be used to texture synthetic images
 - Metadata from labels to inform automatic sythetic image generation

'''

import numpy as np
from tifffile import imread, imwrite
import pickle
from tqdm import tqdm

import sys
import os
import argparse
import glob
import re

parser = argparse.ArgumentParser(description='Sample Real Fluorecence for Synthetic Images.')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--data-folder', type=str, required=True,
                    help='Subfolder name inside data-root containing the images (e.g., Fluo-C3DH-H157-train/).')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()

assert args.data_root[-1] == '/'
assert args.data_folder[-1] == '/'

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
if args.data_root not in sys.path:
    (sys.path).append(args.data_root)

from synthetic_generator import fluorescent_sampler
from utils import load_config

if args.global_config is None:
    global_params = load_config(f'config/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']

disable_labels = global_params['SAMPLER']['DISABLE_LABELS']
use_silver_truth = global_params['SAMPLER']['SILVER_TRUTH']
normalise = global_params['SAMPLER']['NORMALISE']
min_dist = global_params['SAMPLER']['MIN_DIST']
max_dist = global_params['SAMPLER']['MAX_DIST']
dx = global_params['SAMPLER']['DX']
skip = global_params['SAMPLER']['SKIP']

# print(f'Region Width: {dx:.3f}' + r'\mu m')
# print(f'Range: ({min_dist:.3f}, {max_dist:.3f})')
# print(f'Skip {skip}')

data_path = args.data_root + args.data_folder

## Main Code

# get labels
if use_silver_truth:
    label_type = 'ST'
else:
    label_type = 'GT'

# find all labelled images
man_seg_files = sorted(glob.glob(f'{data_path}*_{label_type}/SEG/man_seg*.tif'))
N = len(man_seg_files)

os.makedirs(f'data_generator/sampled_data/', exist_ok=True)
for n in tqdm(range(N), desc='Sampling fluorescence'):
    
    f = man_seg_files[n]

    # If seg file suggests if the labels are volume (N_values=1) or slice (N_values=2)
    N_values = len(re.findall(r'\d+', f.split('/')[-1]))

    if N_values == 1: ## We have volume labels:
        # print('Volume Labels')

        cell, t_lab = re.findall(r'\d+', f)[-2:]
        tp = int(t_lab[:2].lstrip('0') + t_lab[-1]) # timepoint

        # load data
        image = imread(f'{data_path}{cell}/t{t_lab}.tif')
        try:
            mask = imread(f'{data_path}{cell}_{label_type}/SEG/man_seg{t_lab}.tif') # integer array for multiple cells
        except:
            mask = imread(f'{data_path}{cell}_{label_type}/SEG/man_seg_{t_lab}.tif') # integer array for multiple cells

        if normalise:
            image = (image-image.mean())/image.std()

        # # ensure labelling 0, 1, 2 .. with no gaps!
        # labelled_mask = np.zeros_like(mask)
        # for i, lab in enumerate(np.unique(mask)):
        #     labelled_mask[mask==lab] = i

        # sampler file name
        save_path = f'data_generator/sampled_data/data_{args.dataset_id}/{cell}_t{t_lab}.pkl'
        os.makedirs(f'data_generator/sampled_data/data_{args.dataset_id}/', exist_ok=True)

        # sample fluorescence
        sampler = fluorescent_sampler(image, 
                                      (labelled_mask>.5),
                                      labelled_mask=mask,
                                      sampling=sampling,
                                      dx=dx,
                                      min_dist=min_dist,
                                      max_dist=max_dist,
                                      save_path=save_path,
                                      skip=skip,
                                      disable_labels=disable_labels,
                                      jitter_sigma=0,
                                      bg_partition=False # keep the background region as one region
                                      )

        
    elif N_values == 2: ## We have slice labels:

        cell, t_lab, zs_lab = re.findall(r'\d+', f)[-3:]
        tp = int(t_lab[:2].lstrip('0') + t_lab[-1]) # timepoint
        zs = int(zs_lab[:2].lstrip('0') + zs_lab[-1]) # z slice

        # load data
        image = imread(f'{data_path}{cell}/t{t_lab}.tif')[zs]
        try:
            mask = imread(f'{data_path}{cell}_{label_type}/SEG/man_seg{t_lab}_{zs_lab}.tif') # integer array for multiple cells
        except:
            mask = imread(f'{data_path}{cell}_{label_type}/SEG/man_seg_{t_lab}_{zs_lab}.tif') # integer array for multiple cells

        if normalise:
            image = (image-image.mean())/image.std()

        # # ensure labelling 0, 1, 2 .. with no gaps!
        # labelled_mask = np.zeros_like(mask)
        # for i, lab in enumerate(np.unique(mask)):
        #     labelled_mask[mask==lab] = i


        # sampler file name
        save_path = f'data_generator/sampled_data/data_{args.dataset_id}/{cell}_t{t_lab}_z{zs_lab}.pkl'
        os.makedirs(f'data_generator/sampled_data/data_{args.dataset_id}/', exist_ok=True)

        # sample fluorescence
        sampler = fluorescent_sampler(image[np.newaxis], 
                                      (mask>.5)[np.newaxis],
                                      labelled_mask=mask[np.newaxis],
                                      sampling=sampling,
                                      dx=dx,
                                      min_dist=min_dist,
                                      max_dist=max_dist,
                                      save_path=save_path,
                                      skip=skip,
                                      disable_labels=disable_labels,
                                      bg_partition=False # keep the background region as one region
                                     )
        

    else: # Not a valid file format
        raise Exception('Label Files not Valid. \n Must be man_seg000.tif for volume labels, or man_seg000_000.tif for slice labels.')