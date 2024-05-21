#!/usr/bin/env python
# coding: utf-8

# LOAD LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import skimage.util
from typing import Any, Optional, Tuple
import math
import spect.nmr_timefit as fit

# FUNCTIONS TO CALL INSIDE MAIN FUNCTION
def round_up(x: float, decimals: int = 0) -> float:
    return math.ceil(x * 10**decimals) / 10**decimals
 
def calculate_static_spectroscopy(
    fid: np.ndarray,
    dwell_time: float = 1.95e-05,
    tr: float = 0.015,
    center_freq: float = 34.09,
    rf_excitation: int = 218,
    n_avg: Optional[int] = None,
    n_avg_seconds: int = 1,
    method: str = "voigt",
    plot: bool = False,
) -> Tuple[float, Any]:
    """Fit static spectroscopy data to Voigt model and extract RBC:M ratio.

    The RBC:M ratio is defined as the ratio of the fitted RBC peak area to the membrane
    peak area.
    Args:
        fid (np.ndarray): Dissolved phase FIDs in format (n_points, n_frames).
        dwell_time (float): Dwell time in seconds.
        tr (float): TR in seconds.
        center_freq (float): Center frequency in MHz.
        rf_excitation (int, optional): _description_. Excitation frequency in ppm.
        n_avg (int, optional): Number of FIDs to average for static spectroscopy.
        n_avg_seconds (int): Number of seconds to average for
            static spectroscopy.
        plot (bool, optional): Plot the fit. Defaults to False.

    Returns:
        Tuple of RBC:M ratio and fit object
    """
    t = np.array(range(0, np.shape(fid)[0])) * dwell_time
    t_tr = np.array(range(0, np.shape(fid)[1])) * tr
    start_time=2
    end_time=10
    start_ind = np.argwhere(np.array([round_up(x, 2) for x in t_tr]) == start_time)
    end_ind = np.argwhere(np.array([round_up(x, 2) for x in t_tr]) == end_time)
    if np.size(start_ind) == 0:
        start_ind = [0]
    if np.size(end_ind) == 0:
        end_ind = [np.size(t_tr)]
    start_ind= int(start_ind[int(np.floor(np.size(start_ind) / 2))])
    # calculate number of FIDs to average
    if n_avg:
        n_avg = n_avg
    else:
        n_avg = int(n_avg_seconds / tr)
    end_ind = np.min([len(fid[0, :]) - 1, start_ind + n_avg + 1])
    data_dis_avg = np.average(fid[:, start_ind:end_ind], axis=1)
    fit_obj = fit.NMR_TimeFit(
        ydata=data_dis_avg,
        tdata=t,
        area=np.array([1,1,1]),
        freq= np.array([0, -21.7, -218.0]) * center_freq,
        fwhmL=np.array([8.8, 5.0, 2.0]) * center_freq,
        fwhmG=np.array([0, 6.1, 0]) * center_freq,
        phase=np.array([0, 0, 0]),
        line_broadening=0,
        zeropad_size=np.size(t),
        method=method,
    )
    lb = np.stack(
        (
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf, -np.inf],
        )
    ).flatten()
    ub = np.stack(
        (
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
            [+np.inf, +np.inf, +np.inf],
        )
    ).flatten()
    bounds = (lb, ub)
    fit_obj.fit_time_signal_residual(bounds=bounds)
    if plot:
        fit_obj.plot_time_spect_fit()
    rbc_m_ratio = fit_obj.area[0] / np.sum(fit_obj.area[1])
    return rbc_m_ratio, fit_obj
    
def normalize_images(images):
    # initial zero ndarray
    normalized_images = np.zeros_like(images.astype(float))
    # first  index is number of images
    #other indices indicates height, width, depth of the image
    num_images = images.shape[0]
    # computing the minimum and maximum value of the input image for normalization 
    maximum_value, minimum_value = images.max(), images.min()
    # normalize all the pixel values of the images to be from 0 to 1
    for img in range(num_images):
        normalized_images[img, ...] = (images[img, ...] - float(minimum_value)) / float(maximum_value - minimum_value)
    return normalized_images

def correct_b0(
        image: np.ndarray, mask: np.ndarray, max_iterations: int = 100
    ) -> np.ndarray:
        index = 0
        meanphase = 1
        while abs(meanphase) > 1e-7:
            index = index + 1
            diffphase = np.angle(image)
            meanphase = np.mean(diffphase[mask])  # type: ignore
            image = np.multiply(image, np.exp(-1j * meanphase))
            if index > max_iterations:
                break
        return np.angle(image)  # type: ignore
    
def display_multiple_slices2(*arrays):
    '''Displays multiple 3D arrays on the same figure using makeSlide'''
    num_arrays = len(arrays)
    # create a figure with subplots for each array
    fig, axes = plt.subplots(1, num_arrays, figsize=(4 * num_arrays, 4))
    for i, A in enumerate(arrays):
        # display each array using the makeSlide function
        #A = A - np.min(A)
        axes[i].imshow(skimage.util.montage([abs(A[:, :, k]) for k in range(0, A.shape[2])], padding_width=1, fill=0))
        if i==0:
            axes[i].set_title('Membrane')
        elif i==1:
            axes[i].set_title('RBC')
    plt.tight_layout()
    plt.show()

# --- MAIN FUNCTION ---
def DixonDecon(dict_dyn, image_gas_highsnr, image_dissolved, data_type='real'):
        # calculating rbc/membrane ratio
        if data_type == 'synthesized':
            rbc_m_ratio = 1.047285 # assuming synthesized lung data made by Thomen 240208
        else:
            rbc_m_ratio, _ = calculate_static_spectroscopy(
                    fid=dict_dyn["fids_dis"],
                    dwell_time=dict_dyn["dwell_time"],
                    tr=dict_dyn["tr"],
                    center_freq=dict_dyn["freq_center"],
                    rf_excitation=dict_dyn["freq_excitation"],
                    plot=False,
                    )
        print(f'RBC/M Ratio: {rbc_m_ratio}')
        
        # create mask with normal threshold method
        imgGasHiSNRNorm=normalize_images(abs(image_gas_highsnr))
        threshold_value=0.5
        mask_hiSNR = abs(imgGasHiSNRNorm) > threshold_value

        # correct B0 inhomogeneity
        diffphase = correct_b0(image_gas_highsnr, mask_hiSNR)
        # calculate phase shift to separate RBC and membrane
        desired_angle = np.arctan2(rbc_m_ratio, 1.0)  # calculated from the flipcal file
        desired_angle = 45*np.pi/180
        current_angle = np.angle(np.sum(image_dissolved[mask_hiSNR > 0]))
        delta_angle = desired_angle - current_angle
        image_dixon = np.multiply(image_dissolved, np.exp(1j * (delta_angle)))
        image_dixon = np.multiply(image_dixon, np.exp(1j * (-diffphase)))
        
        # separate RBC and membrane components
        image_rbc = (
                np.imag(image_dixon)
                if np.mean(np.imag(image_dixon)[mask_hiSNR]) > 0
                else -1 * np.imag(image_dixon)  # type: ignore
            )
        image_membrane = (
                np.real(image_dixon)
                if np.mean(np.real(image_dixon)[mask_hiSNR]) > 0
                else -1 * np.real(image_dixon)  # type: ignore
            )
           
        # display images 
        display_multiple_slices2(image_membrane, image_rbc)
       
        return(image_membrane, image_rbc, mask_hiSNR, diffphase)