'''
Author: Adam Smith 

Updated version for creating sample-based cell images from labelled data.

Supports multi-cell images and alternative background sampling.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import pickle
from collections import defaultdict

# Functions

def convert_sampler(sampler):

    '''
    Original sampler splits data by Voronoi patterns

    This function converts the original by mergin all 
    background data into an independent region 
    '''

    new_sampler = {"all": sampler["all"]}
    
    positive = defaultdict(lambda: {"x0": [], "x1": [], "y": []})
    
    for sampler_id, bins in sampler.items():
        if sampler_id == "all":
            continue
    
        # Keep only negative bins
        new_sampler[sampler_id] = {}
    
        for bin_id, entry in bins.items():
            if int(bin_id) < 0:
                new_sampler[sampler_id][bin_id] = entry
            else:
                positive[bin_id]["x0"].append(entry["x0"])
                positive[bin_id]["x1"].append(entry["x1"])
                positive[bin_id]["y"].append(entry["y"])
    
    # Build the new "0" entry
    new_sampler["0"] = {}
    
    for bin_id, entry in positive.items():
        new_sampler["0"][bin_id] = {
            "x0": min(entry["x0"]),
            "x1": min(entry["x1"]),
            "y": np.concatenate(entry["y"]),
        }
        
    return new_sampler


def custom_resample(values, num_samples=1000):
    '''
    Placeholder for more sophisticated sampling - default is random choice sampling
    '''
    resampled_values = np.random.choice(values, replace=True, size=num_samples)
    
    return resampled_values

def jitter_partition_map(partitions, labels, sigma):
    """
    Jitter a partition map by sampling random voxel-wise displacements
    from Gaussian(s). Inside-cell voxels remain fixed.
    
    Parameters
    ----------
    partitions: ndarray, shape (Z,X,Y)
        Integer label map for nearest cell.
    labels : ndarray, shape (Z,X,Y)
        Integer label map (0=background, 1+=cell labels).
    sigma : float or sequence of 3 floats
        Standard deviation(s) for displacement sampling (in voxels).
        - float: isotropic jitter
        - (sigma_z, sigma_x, sigma_y): anisotropic jitter
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    jittered : ndarray, shape (Z,X,Y)
        Partition map with jittered background/partition boundaries.
    """
    
    assert partitions.shape == labels.shape
    
    Z, X, Y = partitions.shape
    
    # normalize sigma input
    if np.isscalar(sigma):
        sigma_z, sigma_x, sigma_y = sigma, sigma, sigma
    else:
        if len(sigma) != 3:
            raise ValueError("sigma must be a float or a sequence of 3 floats (z, x, y)")
        sigma_z, sigma_x, sigma_y = sigma

    # coordinate grid
    zz, xx, yy = np.meshgrid(np.arange(Z), np.arange(X), np.arange(Y),
                             indexing="ij", sparse=False)
    
    # random displacements (rounded to nearest int)
    dz = np.random.normal(0, sigma_z, size=partitions.shape).round().astype(int)
    dx = np.random.normal(0, sigma_x, size=partitions.shape).round().astype(int)
    dy = np.random.normal(0, sigma_y, size=partitions.shape).round().astype(int)
    
    # keep interior of cells fixed
    mask_interior = labels > 0
    dz[mask_interior] = 0
    dx[mask_interior] = 0
    dy[mask_interior] = 0
    
    # displaced coordinates, clamped
    z_new = np.clip(zz + dz, 0, Z-1)
    x_new = np.clip(xx + dx, 0, X-1)
    y_new = np.clip(yy + dy, 0, Y-1)
    
    # gather reassigned partitions
    jittered = partitions[z_new, x_new, y_new]
    
    return jittered


# Sampler 
def fluorescent_sampler(image, 
                        instances,
                        sampling=(1.0,1.0,1.0),
                        dx=2,
                        min_dist=-50,
                        max_dist=50,
                        save_path=None,
                        subsample=5,
                        disable_labels=False,
                       ):

    '''
    Fluorescent Sampler for generating sample-based synthetic cell images.

    Converts labelled images/volumes into 'sampler' for image generator. 2D and 3D input supported!

    image: array source of intensities used to sample.
    instances: bool or int array of image labels.
    sampling: pixel/voxel size used to calculate distance maps.
    dx: value interval width for sampling.
    min_dist, max_dist: values that define the range of sampling.
    subsample: int, rate to subsample intensity values in defined intervals.
    save_path: Optional, path to save sampler.

    Note: The range [min_dist, max_dist] with interval width 'dx' must include 0.
    
    '''

    # inputs conditions
    assert image.shape == instances.shape 
    
    assert (np.issubdtype(instances.dtype, np.bool_) or 
            np.issubdtype(instances.dtype, np.integer))

    # interval conditions
    assert min_dist <= 0
    assert max_dist >= 0
    
    assert 0 in np.arange(min_dist, max_dist+1e-5, dx) 

    # bin parameters
    min_bins = 5 # minimum number of intervals per instance
    min_bin_size = 10 # min values in a single bin
    
    # bool instances are supported, but need to convert int
    if disable_labels:
        ''' 
        Remove instance labelling. Pools the intensity values from all cells into one label

        The is suitable when:
            1. The images have a lot of labelled cells
            2. Intensity variation across cells is very low
        '''
        instances = (instances>.0).astype(int)
    else:
        instances = instances.astype(int)
    
    # Calculate distance map and image partition
    dist_map, indices = distance_transform_edt(instances==0, sampling=sampling, return_indices=True)
    for j in range(instances.max()):
        dist_map -= distance_transform_edt(instances==j+1, sampling=sampling)
        
    partitions = instances[tuple(indices)]
    
    ## bin the values by distance
    x = dist_map.flatten()[::subsample]
    y = image.flatten()[::subsample]
    part_flat = partitions.flatten()[::subsample]
    
    y = y[(min_dist < x)&(x < max_dist)]
    part_flat = part_flat[(min_dist < x)&(x < max_dist)]
    x = x[(min_dist < x)&(x < max_dist)]

    # define intervals and the corresponding labels
    intervals = np.arange(min_dist, max_dist+1e-5, dx)
    
    l = 0; interval_lab = []
    for x0, x1 in zip(intervals[:-1], intervals[1:]):
        l+=1
        if x0 == 0:
            v = 1*l; l+=1
        interval_lab.append(l)
    interval_lab = np.array(interval_lab) - v

    # compute output
    output = {'all':{'x':x,'y':y}}
    lab_counter = 1
    for lab in range(1,instances.max()+1):
        binned_data = {}
        initialised = False
        ended = False
        
        for l, x0, x1 in zip(interval_lab, intervals[:-1], intervals[1:]):
            if len(y[(x0<x)&(x<=x1)&(part_flat==lab)]) > min_bin_size: # Trim
                binned_data[str(l)] = {'x0':x0, 'x1':x1, 'y': y[(x0<x)&(x<=x1)&(part_flat==lab)]}

        successful_bins = [s for s in binned_data]
        
        # Check continuity
        continuous = True
        for b0, b1 in zip(successful_bins[:-1], successful_bins[1:]):
            if binned_data[b0]['x1'] != binned_data[b1]['x0']:
                continuous = False

        if not continuous:
            print(f'Label {lab} not continuous.')

        if len(successful_bins) < min_bins:
            print(f'Label {lab} too small.')

        if continuous and len(successful_bins) >= min_bins:
            
            # only add the extended internal boundaries if at least one internal bin exists
            if int(successful_bins[0]) < 0:
                binned_data[successful_bins[0]]['x0'] = -1000
                
            # only add the extended external boundaries if at least one external bin exists
            if int(successful_bins[-1]) > 0:
                binned_data[successful_bins[-1]]['x1'] = 1000
            
            output[str(lab)] = binned_data
                            
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(output, f)

    return output


# Image Generator
def texture_mask(instances, 
                 sampler, 
                 sampling=(1.0,1.0,1.0), 
                 dist_map=None, 
                 dm_noise=None, 
                 vm_noise=None,
                 fix_instances=False):

    '''
    Generate Sample-based cell images with texture_mask

    Takes sampler obj. from fluorescent_sampler and generates textured image with instances provided.

    instances: bool or int array of image labels used to guide sampling.
    sampler: obj. saved from fluorescent_sampler
    sampling: pixel/voxel size used to calculate distance maps. Can be different to source img.
    dist_map: float array, signed distance map of instances (negative inside).
    dm_noise: float, standard deviation of additive gaussian noise for distance map. Reduces block-y 
              appearance. If None, no noise added.
    vm_noise: float, standard deviation of additive gaussian noise for voronoi map. Reduces block-y 
              appearance between cells. If None, no noise added.
    
    '''

    # check inputs
    assert (np.issubdtype(instances.dtype, np.bool_) or 
            np.issubdtype(instances.dtype, np.integer))
    if dist_map is not None:
        assert dist_map.shape == instances.shape        
    if dm_noise is not None:
        assert dm_noise >= 0
    if vm_noise is not None:
        assert vm_noise >= 0
    
    if dist_map is None: # calculate distance map and voronoi partition if not provided
        # Calculate the distance map
        dist_map, indices = distance_transform_edt(instances==0, sampling=sampling, return_indices=True)
        for j in range(instances.max()):
            dist_map -= distance_transform_edt(instances==j+1, sampling=sampling)
        voronoi = instances[tuple(indices)]
        
    else: # just get the indices for voronoi partition
        _, indices = distance_transform_edt(instances==0, sampling=sampling, return_indices=True)
        
    if vm_noise is not None: # use voronoi partition
        voronoi = instances[tuple(indices)]
    else: # treat background separate
        voronoi = instances
        
    if voronoi.max() > 1:
        if vm_noise is not None: # add noise to voronoi pattern
            voronoi = jitter_partition_map(voronoi, instances, vm_noise)
        else: # merge background
            sampler = convert_sampler(sampler)
            
    # update instance labels to match labels in sampler
    if fix_instances:
        instances_updated = instances
        voronoi_updated = voronoi
        new_labels = np.unique(instances)[1:]
    else:
        all_cell_labels = [int(k) for k in sampler if k not in ['all','0']]
        instances_updated = np.zeros_like(instances)
        voronoi_updated = np.zeros_like(voronoi)
        new_labels = []
        for lab in range(1,voronoi.max()+1):
            sampled_lab = np.random.choice(all_cell_labels)        
            instances_updated[instances==lab] = sampled_lab # change label
            voronoi_updated[voronoi==lab] = sampled_lab # change label        
            new_labels.append(sampled_lab)
    
    # Image Array
    output_image = np.zeros_like(dist_map)

    if dm_noise is not None: ## blur but keep positive and negative separate
        dist_map += np.random.normal(loc=0, scale=dm_noise, size=dist_map.shape) 

    # label_regions = [int(lab) for lab in sampler if lab!='all'] # list of available regions in sampler
    
    for lab in np.unique(voronoi_updated): # if a label matches on the synthetic image, sample values
        
        for s in sampler[str(lab)]:
            
            lb = sampler[str(lab)][s]['x0']; ub = sampler[str(lab)][s]['x1']
            
            voxels = (lb<dist_map)&(dist_map<=ub)&(voronoi_updated==lab) # pixels within the boundary
            
            if len(voxels) > 0:
                N = np.sum(voxels)

                # sample fluorescence from real images
                sampled_vals = custom_resample(sampler[str(lab)][s]['y'], num_samples=N)
            
                # Get the 3D coordinates of the voxels
                voxel_coords = np.where(voxels)
            
                # Assign resampled values directly
                output_image[voxel_coords] = sampled_vals
        
    return output_image