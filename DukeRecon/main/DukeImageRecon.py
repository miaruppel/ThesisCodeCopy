#!/usr/bin/env python
# coding: utf-8

# LOAD LIBRARIES
import numpy as np
import time
from recon import dcf, kernel, proximity, recon_model, system_model
import matplotlib.pyplot as plt
import skimage.util

# FUNCTIONS TO CALL INSIDE MAIN FUNCTION
# Duke's recon function
def reconstruct(
    data: np.ndarray,
    traj: np.ndarray,
    kernel_sharpness: float = 0.32,
    kernel_extent: float = 0.32 * 9,
    overgrid_factor: int = 3,
    image_size: int = 128,
    n_dcf_iter: int = 20,
    verbosity: bool = True,
) -> np.ndarray:
    """Reconstruct k-space data and trajectory.
    Args:
        data (np.ndarray): k space data of shape (K, 1)
        traj (np.ndarray): k space trajectory of shape (K, 3)
        kernel_sharpness (float): kernel sharpness. larger kernel sharpness is sharper
            image
        kernel_extent (float): kernel extent.
        overgrid_factor (int): overgridding factor
        image_size (int): target reconstructed image size
            (image_size, image_size, image_size)
        n_pipe_iter (int): number of dcf iterations
        verbosity (bool): Log output messages
    Returns:
        np.ndarray: reconstructed image volume
    """
    start_time = time.time()
    prox_obj = proximity.L2Proximity(
        kernel_obj=kernel.Gaussian(
            kernel_extent=kernel_extent,
            kernel_sigma=kernel_sharpness,
            verbosity=verbosity,
        ),
        verbosity=verbosity,
    )
    system_obj = system_model.MatrixSystemModel(
        proximity_obj=prox_obj,
        overgrid_factor=overgrid_factor,
        image_size=np.array([image_size, image_size, image_size]),
        traj=traj,
        verbosity=verbosity,
    )
    dcf_obj = dcf.IterativeDCF(
        system_obj=system_obj, dcf_iterations=n_dcf_iter, verbosity=verbosity
    )
    recon_obj = recon_model.LSQgridded(
        system_obj=system_obj, dcf_obj=dcf_obj, verbosity=verbosity
    )
    image = recon_obj.reconstruct(data=data, traj=traj)
    del recon_obj, dcf_obj, system_obj, prox_obj
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: {:.2f} seconds".format(execution_time))
    return image

def flip_and_rotate_image(
    image: np.ndarray
) -> np.ndarray:
    image = np.rot90(np.rot90(image, 3, axes=(1, 2)), 1, axes=(0, 2))
    image = np.rot90(image, 1, axes=(0, 1))
    image = np.flip(np.flip(image, axis=1), axis=2)
    return image
    
def display_multiple_slices1(*arrays):
    '''Displays multiple 3D arrays on the same figure using makeSlide'''
    num_arrays = len(arrays)
    # create a figure with subplots for each array
    fig, axes = plt.subplots(1, num_arrays, figsize=(4 * num_arrays, 4))
    for i, A in enumerate(arrays):
        # display each array using the makeSlide function
        axes[i].imshow(skimage.util.montage([abs(A[:, :, k]) for k in range(0, A.shape[2])], padding_width=1, fill=0))
        if i==0:
            axes[i].set_title('Gas - High Resolution')
        elif i==1:
            axes[i].set_title('Gas - High SNR')
    plt.tight_layout()
    plt.show()


# --- MAIN FUNCTION ---
def ImageRecon(phasors_128, trajlist, data_type='real'):
    # GAS RECON
    # running main recon function
    image_gas_highsnr = reconstruct(
    data=phasors_128,
    traj=trajlist,
    kernel_sharpness=float(0.14),
    kernel_extent=9 * float(0.14),
    image_size=int(128),
    )

    image_gas_highreso = reconstruct(
    data=phasors_128,
    traj=trajlist,
    kernel_sharpness=float(0.32),
    kernel_extent=9 * float(0.32),
    image_size=int(128),
    ) 

    # DISSOLVED RECON
    image_dissolved = reconstruct(
            data=phasors_128,
            traj=trajlist,
            kernel_sharpness=float(0.14),
            kernel_extent=9 * float(0.14),
            image_size=int(128),
           )
    
    # displaying based on real or synthesized data
    if data_type == 'synthesized': # need to swap axes and reverse order to get back to original image for synthesized data
        image_gas_highreso_altered = np.transpose(image_gas_highreso, (1, 2, 0))[127::-1, 127::-1, 127::-1]
        image_gas_highsnr_altered = np.transpose(image_gas_highsnr, (1, 2, 0))[127::-1, 127::-1, 127::-1]
        image_dissolved_altered = np.transpose(image_dissolved, (1, 2, 0))[127::-1, 127::-1, 127::-1]
        
        display_multiple_slices1(np.abs(image_gas_highreso_altered), np.abs(image_gas_highsnr_altered))
        
        return (image_gas_highreso_altered, image_gas_highsnr_altered, image_dissolved_altered)
    
    else: # real data (default)
        image_gas_highreso = flip_and_rotate_image(image_gas_highreso)  
        image_gas_highsnr = flip_and_rotate_image(image_gas_highsnr)
        image_dissolved = flip_and_rotate_image(image_dissolved)
        
        display_multiple_slices1(np.abs(image_gas_highreso), np.abs(image_gas_highsnr))
        
        return (image_gas_highreso, image_gas_highsnr, image_dissolved)