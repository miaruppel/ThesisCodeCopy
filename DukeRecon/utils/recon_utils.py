"""Reconstruction util functions."""
import sys

sys.path.append("..")
from typing import Tuple
import pdb
import numpy as np


def get_noisy_projections(
    data: np.ndarray,
    snr_threshold: float = 0.7,
    tail: float = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove noisy FID rays in the k space data by finding indices mask.

    Remove noisy FIDs in the kspace data and their corresponding trajectories.

    Args:
        data (np.ndarray): k space datadata of shape (n_projections, n_points)
        thre_snr (float): threshold SNR value
        tail (float, optional): Index to define the tail of FID. Defaults to 10.

    Returns:
        Returns a boolean mask of the indices of the good FIDs.
    """
    thre_dis = snr_threshold * np.average(abs(data[:, :5]))
    max_tail = np.amax(abs(data[:, tail:]), axis=1)
    return max_tail < thre_dis


def apply_indices_mask(
    data: np.ndarray,
    traj: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply indices mask to data and trajectory.

    Args:
        data (np.ndarray): k space datadata of shape (n_projections, n_points)
        traj (np.ndarray): trajectory of shape (n_projections, n_points, 3)
        indices (np.ndarray): boolean mask of indices to keep.

    Returns:
        Tuple of the data, and traj coordinates with the noisy FIDs removed
        given by the indices mask.
    """
    return (data[indices], traj[indices])


def flatten_data(data: np.ndarray) -> np.ndarray:
    """Flatten data for reconstruction.

    Args:
        data (np.ndarray): data of shape (n_projections, n_points)

    Returns:
        np.ndarray: flattened data of shape (n_projections * n_points, 1)
    """
    return data.reshape((data.shape[0] * data.shape[1], 1))


def flatten_traj(traj: np.ndarray) -> np.ndarray:
    """Flatten trajectory for reconstruction.

    Args:
        traj (np.ndarray): trajectory of shape (n_projections, n_points, 3)
    Returns:
        np.ndarray: flattened trajectory of shape (n_projections * n_points, 3)
    """
    return traj.reshape((traj.shape[0] * traj.shape[1], 3))
