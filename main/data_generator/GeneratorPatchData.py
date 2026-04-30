import numpy as np
import os
from tqdm import tqdm
import pickle
import sys
import argparse
import glob
from tifffile import imread, imwrite

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

from scipy import ndimage
from skimage.measure import regionprops, label
from data_generator.synthetic_generator import texture_mask
from utils import load_config, setup_logger

parser = argparse.ArgumentParser(description='Generate Synthetic Data for 3D Extrusion Event Detection.')
parser.add_argument('--N', type=int, required=True,
                    help='Number of Samples to Generate')
parser.add_argument('--sampler-dir', type=str, required=True,
                    help='Directory with Sampled Fluorescence')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Output Folder')
parser.add_argument('--synth-config', type=str, default=None,
                    help='Config. File for Synth Data')
parser.add_argument('--global-config', type=str, default=None,
                    help='Global Config. File for Dataset')
parser.add_argument('--logger', type=str, default='log.txt',
                    help='.txt file for Logging. Recommend for each sytnthetic dataset.')
args = parser.parse_args()

''' FUNCTIONS '''

def unique_count_filt(arr, window_size):
    """
    Count # unique values in slicing window
    """
    if isinstance(window_size, int):
        size = (window_size, window_size)
    else:
        size = window_size

    return ndimage.generic_filter(
        arr,
        function=lambda x: len(np.unique(x)),
        size=size,
        mode="nearest"
    )
    
def rosette_features_xyz(labelled_volume, rosette_labels):

    rosette_xyz = np.zeros_like(labelled_volume)
    for lab in rosette_labels:
        rosette_xyz += lab*(labelled_volume==lab).astype(int)

    rosette_cell_radii = []
    for R in regionprops(rosette_xyz):
        rosette_cell_radii.append( (R.area*.755*.755*1.384*(3/(4*np.pi)))**(1/3) )
    rosette_cell_radii = np.array(rosette_cell_radii)

    # rosette not at the edge of the image
    z_lb = (np.sum(rosette_xyz[0,:,:]) == 0); z_ub = (np.sum(rosette_xyz[-1,:,:]) == 0);
    x_lb = (np.sum(rosette_xyz[:,0,:]) == 0); x_ub = (np.sum(rosette_xyz[:,-1,:]) == 0);
    y_lb = (np.sum(rosette_xyz[:,:,0]) == 0); y_ub = (np.sum(rosette_xyz[:,:,-1]) == 0);

    edge_criteria = np.array([z_lb,z_ub,x_lb,x_ub,y_lb,y_ub]).all()

    return rosette_cell_radii, edge_criteria

def rotate_label_patch_aniso(patch, spacing, angle_zx, angle_zy, pad_value=0):
    """
    Rotate a label patch in physical space while preserving labels.
    """

    sz, sx, sy = spacing

    # --- Rotation matrices ---
    theta = np.deg2rad(angle_zx)
    phi   = np.deg2rad(angle_zy)

    # rotate in zx plane (around y)
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # rotate in zy plane (around x)
    Rx = np.array([
        [1, 0,              0             ],
        [0, np.cos(phi),   -np.sin(phi)],
        [0, np.sin(phi),    np.cos(phi)]
    ])

    R = Rx @ Ry

    # spacing matrix
    S = np.diag([sz, sx, sy])

    # full transform in voxel coordinates
    M = np.linalg.inv(S) @ R @ S

    # center rotation
    center = 0.5 * np.array(patch.shape)
    offset = center - M @ center

    rotated = ndimage.affine_transform(
        patch,
        matrix=M,
        offset=offset,
        order=0,           # 0 nearest neighbour for labels
        mode='constant',
        cval=pad_value
    )

    return rotated


def extract_rotated_patches_aniso( volume, centers, spacing=(1,1,1), patch_size=(16,64,64), p=0.5, angle_range=(-30,30), pad_value=0, interpolation_order=1):
    """
    Extract anisotropy-aware rotated patches.

    Parameters
    ----------
    volume : ndarray OR list of ndarray
        Either:
            - single 3D array
            - list of aligned 3D arrays

        If a list is given:
            - identical extraction + augmentation is applied to all arrays
            - useful for image + labels + distance maps etc.

    centers : list of (z,x,y)
    spacing : (sz,sx,sy)
    patch_size : output patch size
    p : probability of rotation
    angle_range : rotation range in degrees
    pad_value : constant padding value
    interpolation_order :
        0 = nearest neighbour (labels)
        1 = linear interpolation (images)

    Returns
    -------
    patches :
        if single volume:
            ndarray (N, dz, dx, dy)
        if list:
            list of ndarrays
            one output array per input volume
    """

    # normalize input
    single_input = not isinstance(volume, (list, tuple))

    if single_input:
        volumes = [volume]
    else:
        volumes = list(volume)

    # check shapes
    shape = volumes[0].shape
    for v in volumes:
        assert v.shape == shape

    Z, X, Y = shape

    # geometry
    sz, sx, sy = spacing

    dz, dx, dy = patch_size
    hz, hx, hy = dz//2, dx//2, dy//2

    phys_radius = np.sqrt((dx*sx)**2 + (dy*sy)**2) / 2

    margin_z = int(np.ceil(phys_radius / sz))
    margin_x = int(np.ceil(phys_radius / sx))
    margin_y = int(np.ceil(phys_radius / sy))

    # output containers
    extracted = [[] for _ in volumes]

    # loop centers
    for cz, cx, cy in centers:

        # SAME random transform for all arrays
        do_rotate = np.random.rand() < p

        if do_rotate:
            a1 = np.random.uniform(*angle_range)
            a2 = np.random.uniform(*angle_range)

        # extract larger patch coordinates
        z0, z1 = cz - hz - margin_z, cz + hz + margin_z
        x0, x1 = cx - hx - margin_x, cx + hx + margin_x
        y0, y1 = cy - hy - margin_y, cy + hy + margin_y

        z0c, z1c = max(0, z0), min(Z, z1)
        x0c, x1c = max(0, x0), min(X, x1)
        y0c, y1c = max(0, y0), min(Y, y1)

        pad_width = ((z0c - z0, z1 - z1c),(x0c - x0, x1 - x1c),(y0c - y0, y1 - y1c))

        # process each aligned volume
        for vi, vol in enumerate(volumes):

            # extract
            patch = vol[z0c:z1c, x0c:x1c, y0c:y1c]

            # pad if needed
            if any(pw[0] > 0 or pw[1] > 0 for pw in pad_width):
                patch = np.pad( patch, pad_width, mode='constant', constant_values=pad_value)

            # isotropic resampling
            iso_patch = ndimage.zoom( patch, (sz/sx, 1, sy/sx), order=interpolation_order)

            # rotation
            if do_rotate:
                iso_patch = rotate_label_patch_aniso( iso_patch, spacing, a1, a2, pad_value=pad_value)

            # back to original spacing
            patch = ndimage.zoom( iso_patch, (sx/sz, 1, sx/sy), order=interpolation_order)

            # crop center
            bz, bx, by = patch.shape
            cz2, cx2, cy2 = bz//2, bx//2, by//2

            final = patch[cz2-hz:cz2+hz,cx2-hx:cx2+hx,cy2-hy:cy2+hy]

            extracted[vi].append(final)

    # stack outputs
    extracted = [np.stack(p) for p in extracted]

    if single_input:
        return extracted[0]

    return extracted

''' PARAMETERS '''

# load configs
if args.synth_config is None:
    synth_config = load_config(f'config/synth_parameters.yaml')
else:
    synth_config = load_config(args.synth_config)

if args.global_config is None:
    global_params = load_config(f'config/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

sampling = global_params['SAMPLING']

zres, xres, yres = synth_config['SHAPE']['IMAGE_SIZE']
r0, r1 = synth_config['SHAPE']['CELL_R'] # 6, 9
z_scale = synth_config['SHAPE']['Z_SCALE']
r_sep = synth_config['SHAPE']['CELL_SEPARATION'] # 2
patch_size = synth_config['SHAPE']['PATCH_SIZE']

distmap_blur = synth_config['SAMPLER']['DISTMAP_BLUR']
distmap_sig = synth_config['SAMPLER']['DISTMAP_SIG']
gaussian_blur = synth_config['SAMPLER']['GAUSSIAN_BLUR']
gaussian_sig = synth_config['SAMPLER']['GAUSSIAN_SIG']

# Non-Config Parameters
N_max = 8000 # cells per layer (keep high to pack shapes)
w_s = 5 # Sliding window size to calculate unique integers in 2D slice
w_z = 1 # width of layer sample space. Increase reduces uniformity, but reduces chance of extrusion (i think)
rosette_size_parameter = 0.7 # the mean radii of rosette cells must exceed this proportion of all synthetic cell radii
extrusion_p = 0.7 # extrusions size as a fraction of mean cell radii (MUST BE >= 0.7)
rotation_p = 0.2 # percentage of patches rotated between -30 and 30 degrees.

scale_peak = 1.75
scale_sigma = 1.5

assert extrusion_p >= 0.7

''' MAIN '''

counter = 0
generation_counter = 0

sampler_files = sorted(glob.glob(f'{args.sampler_dir}*.pkl'))
assert len(sampler_files) > 0

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}control/", exist_ok=True)
os.makedirs(f"{args.output_dir}extrusion/", exist_ok=True)

pbar = tqdm(total=args.N)
logger = setup_logger(log_file=args.logger, clear_log=True)
logger.info("Starting synthetic data generator for 3D extrusion classifier.")
logger.info(f"Samplers Loaded: {len(sampler_files)}")
logger.info(f"Parameters: N_max: {N_max}, w_s: {w_s}, w_z: {w_z}, rosette_size_parameter: {rosette_size_parameter}, extrusion_p: {extrusion_p}, rotation_p: {rotation_p}")

while counter < args.N: # counter for control-extrusion pairs. N number of samples in total
    '''
    Steps:

    1. Generate new packed cell tissue
    2. Along the centre slice of each layer, measure the number of unique integers to identify candidate extrusions
    3. Accept candidate extrusions if conditions satisfied (edge_criteria, rosette size criteria)
    4. Add extrusions with designated size relative to mean cell radii
    5. Check the correct number of extrusions have been added. If not, start again.
    6. Rotate and Cut out extrusion and control patches
    7. Save samples to dataset, update counter!
    '''

    '''
    STEP 1

    1.1: Determine cell centroid locations based on the parameters provided 
    1.2: Generate image labels
    1.3: Compute radii of all cells
    '''

    '''
    Step 1.1
    '''
    Z, X, Y = np.meshgrid(np.arange(0,zres), np.arange(0,xres), np.arange(0,yres), indexing='ij')

    layers = np.arange(4, zres, 8)
    ss_layers = [[z-w_z,z+w_z] for z in layers]

    if generation_counter == 0:
        logger.debug(f"Image Size ({zres},{xres},{yres})")
        logger.debug(f"Cell Layers: {len(layers)}")
    
    centroids = []
    for z0, z1 in ss_layers:
        
        sZ, sX, sY = np.meshgrid(np.arange(z0,z1), np.arange(0,xres), np.arange(0,yres), indexing='ij')
        
        # radius of new cell
        r_c = np.random.uniform(r0,r1)
        
        sample_space = np.ones(sZ.shape, bool)
        
        cell_counter = 0
        while sample_space.sum() > 0:
        
            # all possible coordinates for new cell
            z_space = sZ[sample_space];  x_space = sX[sample_space]; y_space = sY[sample_space]
            coords = np.vstack((z_space,x_space,y_space))
            
            # randomly select a coordinate
            i = np.random.choice(coords.shape[1])         
            z_c = z_space[i];  x_c = x_space[i]; y_c = y_space[i]
        
            centroids.append([z_c, x_c, y_c])
        
            if len(centroids) == 1: # initialise global distance map
                dist = np.sqrt(((Z-z_c)*sampling[0]/z_scale)**2+((X-x_c)*sampling[1])**2+((Y-y_c)*sampling[1])**2) - r_c
            else:
                dist = np.minimum(dist, np.sqrt(((Z-z_c)*sampling[0]/z_scale)**2+((X-x_c)*sampling[1])**2+((Y-y_c)*sampling[1])**2) - r_c)
    
            if cell_counter == 0: # distance map in layer/sample space
                dist_s = np.sqrt(((sZ-z_c)*sampling[0]/z_scale)**2+((sX-x_c)*sampling[1])**2+((sY-y_c)*sampling[1])**2) - r_c
            else:
                dist_s = np.minimum(dist_s, np.sqrt(((sZ-z_c)*sampling[0]/z_scale)**2+((sX-x_c)*sampling[1])**2+((sY-y_c)*sampling[1])**2) - r_c)
        
            cell_counter += 1
        
            # next cell to add
            
            r_c = np.random.uniform(r0,r1)
        
            sample_space = dist_s > r_sep
    
    centroids = np.array(centroids) 

    '''
    Step 1.2
    '''
    
    # interior_tv = -(r0-r_sep+1) # threshold to ensure interior
    interior_tv = -(r0) # threshold to ensure interior
    labelled_interiors = label(dist<=interior_tv)
    
    dist_map, indices = ndimage.distance_transform_edt(labelled_interiors==0, sampling=sampling, return_indices=True)
    
    labelled_volumes = (dist<=3) * labelled_interiors[tuple(indices)] # <=3 dilates cells by 3 ---> packing
    
    '''
    Step 1.3
    '''
    
    cell_radii = []
    for R in regionprops(labelled_volumes):
        cell_radii.append( (R.area*.755*.755*1.384*(3/(4*np.pi)))**(1/3) )
    cell_radii = np.array(cell_radii)
    

    '''
    STEP 2 
    '''

    extrusion_locs = []
    for i, z in enumerate(layers):
        
        #counts
        layer_counts = unique_count_filt(labelled_volumes[z],w_s)
        labelled_counts = label(layer_counts >= 5)
        
        for R in regionprops(labelled_counts):
            x, y = np.array(R.centroid).astype(int)
    
            rosette_labels = np.unique(labelled_volumes[z][x-w_s:x+w_s,y-w_s:y+w_s])
    
            rosette_cell_radii, edge_criteria = rosette_features_xyz(labelled_volumes, rosette_labels)
            mu_r = np.mean(rosette_cell_radii)
            p_r = np.sum(cell_radii<mu_r)/len(cell_radii)
            
            # print(z)
            # print('- Solidity:',solidity)
            # print('- Edge Criteria:',edge_criteria)
            # print(f'- Mean Cell Radii: {np.mean(rosette_cell_radii)} ({:.3f} %)')

            '''
            STEP 3
            '''
            
            if edge_criteria and p_r > rosette_size_parameter:
                
                # extrusion loc mean of all centroids ~ keeps the cell circular
                centroid_xy = []
                for cell_id in rosette_labels:
                    for C in regionprops(1*(labelled_volumes==cell_id)):
                        centroid_xy.append(C.centroid)
                        
                z_hat, x_hat, y_hat = np.mean(centroid_xy, axis=0)

                extrusion_locs.append([z_hat, x_hat, y_hat])

    generation_counter += 1
    if len(extrusion_locs) == 0: # go back to step 1 if no extrusions
        logger.debug(f"Generation {generation_counter}: No Extrusions")
    
    else:
    
        extrusion_locs = np.array(extrusion_locs).astype(int)
        
        '''
        STEP 4
        '''

        mean_R = np.mean(cell_radii)
        p = .7
        
        new_dist = 1*dist
        
        for zX, xX, yX in extrusion_locs:
            new_dist = np.minimum(new_dist, np.sqrt(((Z-zX)*sampling[0]/z_scale)**2+((X-xX)*sampling[1])**2+((Y-yX)*sampling[1])**2) - extrusion_p*mean_R)
        
        # new_interior_tv = -(min(r0, extrusion_p*mean_R)-r_sep+1) # *
        new_interior_tv = -min(r0, extrusion_p*mean_R) # *
        new_labelled_interiors = label(new_dist<=new_interior_tv)
        
        new_dist_map, new_indices = ndimage.distance_transform_edt(new_labelled_interiors==0, sampling=sampling, return_indices=True)
        
        new_labelled_volumes = (new_dist<=3) * new_labelled_interiors[tuple(new_indices)]

        '''
        STEP 5
        '''
        
        if new_labelled_volumes.max() != labelled_volumes.max() + len(extrusion_locs):
            logger.debug(f"Generation {generation_counter}: Error with Adding {len(extrusion_locs)} Extrusions. ({labelled_volumes.max()} -/-> {new_labelled_volumes.max()})")

        else:
            
            '''
            STEP 6

            6.1: Get the same number of control samples as extrusions
            6.2: Scale the membrane intensities of extrudint cells
            6.3: Extract patches + scale intensities
            6.4  Generate textured images
            '''

            # get the same control samples
            '''
            Step 6.1
            '''

            # initialise control sample space
            control_space = np.zeros([zres, yres, xres], bool)
            control_space[8:-8,32:-32,32:-32] = 1 # ignore edge of img

            # remove areas near extrusions from control sample space
            for zX, xX, yX in extrusion_locs:
                control_space = np.minimum(control_space, np.sqrt(((Z-zX)*sampling[0])**2+((X-xX)*sampling[1])**2+((Y-yX)*sampling[1])**2) > 32*sampling[1])
            
            # all possible coordinates for control patch
            z_ctrl = Z[control_space];  x_ctrl = X[control_space]; y_ctrl = Y[control_space]
            ctrl_coords = np.vstack((z_ctrl,x_ctrl,y_ctrl))
            
            ctrl_patch_locs = []
            for _ in extrusion_locs:
                # randomly select a coordinate
                i = np.random.choice(ctrl_coords.shape[1])         
                z_c = z_ctrl[i];  x_c = x_ctrl[i]; y_c = y_ctrl[i]
                
                ctrl_patch_locs.append([z_c, x_c, y_c])
            
            ctrl_patch_locs = np.array(ctrl_patch_locs).astype(int)

            '''
            Step 6.2
            '''

            extrusion_labels = [new_labelled_volumes[pos[0],pos[1],pos[2]] for pos in np.array(extrusion_locs)]

            # signed distance map of extruding cell membranes
            extrusion_membrane_distmap = np.zeros(new_labelled_volumes.shape)
            for ex_label in extrusion_labels:
                extrusion_membrane_distmap -= ndimage.distance_transform_edt(new_labelled_volumes==ex_label, sampling=sampling) # negative inside
            extrusion_membrane_distmap += ndimage.distance_transform_edt(extrusion_membrane_distmap==0, sampling=sampling) # positive outside

            # scalar field, >1 at membrane, 1 everywhere else.
            scaling = 1 + (scale_peak-1)*np.exp(-0.5*(extrusion_membrane_distmap/scale_sigma)**2)

            '''
            Step 6.3
            '''

            # vary loc such that extrusion not always centred
            extrusion_patch_locs = [pos + 0*np.random.normal(loc=[0,0,0],scale=[1,5,5]) for pos in np.array(extrusion_locs)] 
            extrusion_patch_locs = np.array(extrusion_patch_locs).astype(int)

            # ammend patch center such that it is contained in the image
            extrusion_patch_locs = np.maximum(extrusion_patch_locs, np.array(patch_size)//2)
            extrusion_patch_locs = np.minimum(extrusion_patch_locs, np.array([zres,xres,yres]) - np.array(patch_size)//2)

            extrusion_patches, extrusion_patch_scaling = extract_rotated_patches_aniso([new_labelled_volumes, scaling], 
                                                                                       extrusion_patch_locs, 
                                                                                       spacing=sampling, 
                                                                                       patch_size=patch_size, 
                                                                                       p=rotation_p)
            
            ctrl_patches = extract_rotated_patches_aniso(new_labelled_volumes, 
                                                         ctrl_patch_locs, 
                                                         spacing=sampling, 
                                                         patch_size=patch_size, 
                                                         p=rotation_p)

            '''
            Step 6.4
            '''

            control_imgs = []
            extrusion_imgs = []
            
            for patch in ctrl_patches:
                sampler_file = np.random.choice(sampler_files)
                with open(sampler_file, 'rb') as f:
                    sampler = pickle.load(f)
                
                synth_img = texture_mask(patch>0, sampler, 
                                         labelled_mask=patch, 
                                         partitions=None, 
                                         dist_map=None, 
                                         sampling=sampling, 
                                         distmap_blur=distmap_blur, distmap_sig=distmap_sig, 
                                         gaussian_blur=gaussian_blur, gaussian_sig=gaussian_sig)
                control_imgs.append(synth_img)
            
            for patch, scaling in zip(extrusion_patches, extrusion_patch_scaling):
                sampler_file = np.random.choice(sampler_files)
                with open(sampler_file, 'rb') as f:
                    sampler = pickle.load(f)
                
                synth_img = texture_mask(patch>0, sampler, 
                                         labelled_mask=patch, 
                                         partitions=None, 
                                         dist_map=None, 
                                         sampling=sampling,
                                         distmap_blur=distmap_blur, distmap_sig=distmap_sig, 
                                         gaussian_blur=gaussian_blur, gaussian_sig=gaussian_sig)
                extrusion_imgs.append(synth_img * scaling)
        
            '''
            STEP 7
            '''

            for im0, im1 in zip(control_imgs, extrusion_imgs):

                if counter <= args.N: # ensure target reached

                    imwrite(f"{args.output_dir}control/{str(counter//2+1).zfill(3)}.tif", im0)
    
                    imwrite(f"{args.output_dir}extrusion/{str(counter//2+1).zfill(3)}.tif", im1)
    
    
                pbar.update(2)   # tell tqdm one step is done
                counter += 2

            logger.debug(f"Generation {generation_counter}: Success, {len(extrusion_locs)} Extrusions ==> ({counter}/{args.N})")
