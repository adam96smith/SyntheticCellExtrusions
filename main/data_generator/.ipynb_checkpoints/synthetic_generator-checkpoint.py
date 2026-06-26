''' ## NEEDS UPDATING 
Functions to Generate Synthetic Fluorescent Distributions based on real (labelled) data.

*** Part 1: fluorescent sampler ***

Input: image, segmentation (, dist_map, aniso_factor, save_path)
Output: dictionary of organised fluorescent values based on input data.
Parameters:
    dx: distance bin size (default 2)
    max_dist: maximum distance to sample (default 50)

Summary: Organises fluorescent values based on the distance to the segmentation boundary (eg. cell membrane). The fluorescent values are placed in bins based on their distance which can be converted to CDFs for texturing new cells (in Part 2)

*** Part 2: texture function ***

Input: binary mask, input dictionary (from Part 1) (, dist_map, aniso_factor)
Output: synthetic image
Parameters:
    focal_on: bool, (if true, high intensities congregate at focal points at surface)
        min_foci, max_foci: int, number of foci on surface
        min_r, max_r: float, defines size of foci
    distmap_blur: bool, (default True, blurs the distance map to provent the sampler developing blocky distributions)
        distmap_sig: float, (default .1, std of noise added to cause blurring)
    gaussian_blur: bool, (default True, blurs the output image)
        gaussian_sig: float, (default .5, std of blurring)
    resample_fraction: float, (default=.25, fraction of pixels to resample to match texture)

'''

# imports
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label
import pickle


def custom_resample(values, num_samples=1000, bins = 100):
    '''
    Input array of values gets converted to a CDF, and 'num_samples' are drawn
    '''
    # Step 1: Calculate histogram-based probability distribution
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
    pdf = hist / np.sum(hist)  # Normalize to get probability density
    cdf = np.cumsum(pdf)  # Cumulative distribution

    # Step 2: Sample uniformly and map to bin centers
    random_samples = np.random.rand(num_samples)
    sampled_indices = np.searchsorted(cdf, random_samples)
    resampled_values = bin_centers[sampled_indices]  # Map to bin centers
    
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

#### Fluorescent Sampler

def fluorescent_sampler(image, 
                        mask, 
                        labelled_mask=None, 
                        dist_map=None, 
                        partitions=None, 
                        sampling=(1.0,1.0,1.0), 
                        dx=2,
                        min_dist=-50, 
                        max_dist=50, 
                        one_bin_outer=False, 
                        save_path=None,
                        skip=5,
                        disable_labels=False,
                        bg_partition=True,
                       ):

    # check inputs are ok, and generate dist_map if required
    assert image.shape == mask.shape
    assert mask.dtype == bool
    # assert min_dist % dx == 0
    min_bins = 5

    if labelled_mask is None: # create labelled mask if not provided
        labelled_mask = label(mask)
        
    # if dist_map is None or partitions is None: # calculate distance map and image partition if not provided
    #     dist_map, indices = distance_transform_edt(mask==0, sampling=sampling, return_indices=True)
    #     dist_map -= distance_transform_edt(mask, sampling=sampling)

    #     partitions = labelled_mask[tuple(indices)]

    if partitions is None or dist_map is None: # calculate distance map and image partition if not provided
        # Calculate the new full distance map
        dist_map, indices = distance_transform_edt(labelled_mask==0, 
                                                   sampling=sampling, 
                                                   return_indices=True
                                                  )
        for j in range(labelled_mask.max()):
            dist_map -= distance_transform_edt(labelled_mask==j+1, 
                                               sampling=sampling
                                              )
        if bg_partition:
            partitions = labelled_mask[tuple(indices)]
        else:
            partitions = labelled_mask

    assert dist_map.shape == image.shape
    
    ## bin the values by distance
    x = dist_map.flatten()[::skip]
    y = image.flatten()[::skip]
    part_flat = partitions.flatten()[::skip]
    
    y = y[(min_dist < x)&(x < max_dist)]
    part_flat = part_flat[(min_dist < x)&(x < max_dist)]
    x = x[(min_dist < x)&(x < max_dist)]

    if one_bin_outer:
        bins = np.concatenate( (np.arange(min_dist, 0, dx), np.arange(0, max_dist+1e-5, max_dist)))
    else:
        bins = np.concatenate( (np.arange(min_dist, 0, dx), np.arange(0, max_dist+1e-5, dx)))
    
    output = {'all':{'x':x,'y':y}}
    lab_counter = 1

    # if disable_labels:
    #     bin_data = {}
    #     c = 0

    #     # Continuity required over the bins. 
    #     # The bins start when sufficient data, and end at the first 
    #     # point there is insufficient data (or max_dist is reached)
    #     add_bin = False
        
    #     for x0, x1 in zip(bins[:-1], bins[1:]):
    #         # initialise dictionary entry
    #         tmp = y[(x0<x)&(x<=x1)]

    #         # print(x0, x1, len(tmp))
            
    #         if len(tmp) > 10: # sufficient variability in bin
    #             if c == 0:
    #                 bin_data[str(c)] = {'x0':-1000, 'x1':x1, 'y':tmp}
    #                 add_bin = True # initialise 
    #                 c += 1
                
    #             else:  
    #                 if add_bin == True:
    #                     if x1 == max_dist:
    #                         bin_data[str(c)] = {'x0':x0, 'x1':1000, 'y':tmp}
    #                     else:
    #                         bin_data[str(c)] = {'x0':x0, 'x1':x1, 'y':tmp}
    #                     c += 1
    #         else:
    #             # print(f'Insufficient data in ({x0},{x1}) bin')
    #             add_bin = False

    #     print(f'Total, Counter: {c}.')

    #     if c > 3: # at least 3 regions can be sampled
    #         # set last bin  
    #         bin_data[str(c-1)]['x1'] = 1000
    
    #         # Criteria for cell inclusion in sampler
    #         if bin_data[str(c-1)]['x0'] >= 0: # if the bin criteria reaches edge the cell (consider packed cell case)
    #             output[str(lab_counter)] = bin_data
    #             lab_counter += 1
    #     else:
    #         print(f'Image not included in sampler')

    # else:    
    
    l0 = 1 if bg_partition else 0
    for lab in range(l0,labelled_mask.max()+1):
        # print(f'- Label = {lab} (skip={skip})')
        all_bin_data = {}
        initialised = False
        ended = False
        c = 0

        for x0, x1 in zip(bins[:-1], bins[1:]):
            c += 1
            all_bin_data[str(c)] = {'x0':x0, 'x1':x1, 'y': y[(x0<x)&(x<=x1)&(part_flat==lab)]}

        # Trim and check for continuity
        # Trim
        bin_data = {}
        c = 0
        for s in all_bin_data:
            if len(all_bin_data[s]['y']) > 10:
                c += 1
                bin_data[str(c)] = all_bin_data[s]
        
        # Continuity
        continuous = True
        for c0, c1 in zip(np.arange(1,c), np.arange(2,c+1)):
            if bin_data[str(c0)]['x1'] != bin_data[str(c1)]['x0']:
                continuous = False

        # print('-',lab,'-')
        # for s in bin_data:
        #     print(bin_data[s]['x0'], bin_data[s]['x1'], len(bin_data[s]['y']))

        if not continuous:
            print(f'Label {lab} not continuous.')

        if c < min_bins:
            print(f'Label {lab} too small.')

        if continuous and c >= min_bins:
            bin_data['1']['x0'] = -1000
            bin_data[str(c)]['x1'] = 1000
            
            output[str(lab)] = bin_data
                            
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(output, f)

    return output


### Texture Function

def texture_mask(mask, 
                 sampler, 
                 labelled_mask=None, 
                 partitions=None, 
                 dist_map=None, 
                 sampling=(1.0,1.0,1.0), 
                 focal_on=False, 
                 distmap_blur=True, 
                 distmap_sig=.1, 
                 gaussian_blur=True, 
                 gaussian_sig=.5, 
                 resample_fraction=.25,
                 jitter_sigma=5,
                 bg_partition=True,
                ):

    assert mask.dtype == bool
    
    if labelled_mask is None: # create labelled mask if not provided
        labelled_mask = label(mask)
        
    if partitions is None or dist_map is None: # calculate distance map and image partition if not provided
        # Calculate the new full distance map
        dist_map, indices = distance_transform_edt(labelled_mask==0, 
                                                   sampling=sampling, 
                                                   return_indices=True
                                                  )
        for j in range(labelled_mask.max()):
            dist_map -= distance_transform_edt(labelled_mask==j+1, 
                                               sampling=sampling
                                              )

        if bg_partition:
            partitions = labelled_mask[tuple(indices)]
        else:
            partitions = labelled_mask # keeps background as 0

    if partitions.max() > 1 and jitter_sigma > 0:
        partitions = jitter_partition_map(partitions, labelled_mask, jitter_sigma)
    
    all_cell_labels = [int(k) for k in sampler if k not in ['all','0']]
    labelled_mask_updated = np.zeros_like(labelled_mask)
    partitions_updated = np.zeros_like(partitions)
    new_labels = []
    for lab in np.unique(partitions)[1:]:
        sampled_lab = np.random.choice(all_cell_labels)        
        labelled_mask_updated[labelled_mask==lab] = sampled_lab # change label
        partitions_updated[partitions==lab] = sampled_lab # change label        
        new_labels.append(sampled_lab)
    
    # Image Array
    output_image = np.zeros_like(dist_map)

    if distmap_blur: ## blur but keep positive and negative separate
        dist_map += np.random.normal(loc=0, scale=distmap_sig, size=dist_map.shape) 

    label_regions = [int(lab) for lab in sampler if lab not in ['all']] # list of available regions in sampler
    
    for lab in label_regions: # if a label matches on the synthetic image, sample values
        
        for s in sampler[str(lab)]:
            
            lb = sampler[str(lab)][s]['x0']; ub = sampler[str(lab)][s]['x1']
            
            voxels = (lb<dist_map)&(dist_map<=ub)&(partitions_updated==lab) # pixels within the boundary
            
            if len(voxels) > 0:
                N = np.sum(voxels)

                # sample fluorescence from real images
                sampled_vals = custom_resample(sampler[str(lab)][s]['y'], num_samples=N, bins=100)
            
                # Get the 3D coordinates of the voxels
                voxel_coords = np.where(voxels)
            
                # Assign resampled values directly
                output_image[voxel_coords] = sampled_vals

    if gaussian_blur:
        output_image = gaussian_filter(output_image, sigma=gaussian_sig)
        
    return output_image