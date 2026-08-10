from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_err_map(target: np.ndarray,
                 predict: np.ndarray,
                 save_path: str,
                 figsize: tuple = (10, 8),
                 vmin: float = -1.0,
                 vmax: float = 1.0,
                 cbar_shrink: float = 0.7):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0, 'darkblue'), (0.5, 'white'), (1, 'darkred')]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    err_map = target - predict
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(err_map, cmap=cmap, norm=norm)
    # cbar = fig.colorbar(cax, ax=ax, shrink=cbar_shrink, extend='both', extendrect=True)
    ax.axis('off')
    # ax.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_pair(
    target:np.ndarray,
    source:np.ndarray,
    save_path:str,
    figsize=(10,8),
    tgt_cmap:str='Blues',
    src_cmap:str='Oranges',
    src_alpha:float=0.5,
    normalize:bool=True):
    """
    Visualize the paired images
    """
    # At the beginning of the function
    if normalize:
        # Normalize to [0, 1] for better visualization
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        source = (source - source.min()) / (source.max() - source.min() + 1e-8)

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Display images
    im1 = ax.imshow(target, cmap=tgt_cmap, alpha=1.0)
    im2 = ax.imshow(source, cmap=src_cmap, alpha=src_alpha)
    ax.axis('off')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_pair_checker(
    target:np.ndarray,
    source:np.ndarray,
    save_path:str,
    figsize=(10,8),
    stripe_width=25,
):
    """
    Creates a diagonal stripe visualization from two images, similar to the example.

    Args:
        source (np.ndarray): The first image (H, W).
        target (np.ndarray): The second image (H, W).
        stripe_width (int): The width of each diagonal stripe.

    Returns:
        np.ndarray: The combined striped image.
    """
    h, w = source.shape

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Create a mask where stripes are determined by the sum of coordinates
    # The modulo operator creates the repeating pattern
    mask = ((x + y) // stripe_width) % 2 == 0

    # Use the mask to select pixels from either the source or target image
    stripes = np.where(mask, source, target)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(stripes, cmap='gray')
    ax.axis('off')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_pair_w_keypoints(
    target:torch.Tensor,
    source:torch.Tensor,
    tgt_keypnts:torch.Tensor,
    src_keypnts:torch.Tensor,
    select_dim: int,
    select_slice: int,
    save_path:str,
    figsize=(10,8),
    tgt_cmap:str='Blues',
    src_cmap:str='Oranges',
    src_alpha:float=0.5,
    normalize:bool=True):

    target = target.squeeze().cpu().numpy()
    source = source.squeeze().cpu().numpy()

    tgt_keypnts = tgt_keypnts.squeeze().cpu().numpy()
    src_keypnts = src_keypnts.squeeze().cpu().numpy()

    src = np.take(source, indices=select_slice, axis=select_dim)
    tgt = np.take(target, indices=select_slice, axis=select_dim)

    # At the beginning of the function
    if normalize:
        # Normalize to [0, 1] for better visualization
        tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min() + 1e-8)
        src = (src - src.min()) / (src.max() - src.min() + 1e-8)

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Display images
    im1 = ax.imshow(tgt.T, cmap=tgt_cmap, alpha=1.0)
    im2 = ax.imshow(src.T, cmap=src_cmap, alpha=src_alpha)
    ax.axis('off')

    # Get union mask
    if select_dim == 0:
        keypnt_mask = np.logical_and(np.abs(tgt_keypnts[:, 0] - select_slice) <= 1,
                                     np.abs(src_keypnts[:, 0] - select_slice) <= 1)
        tgt_keypnt_slice = tgt_keypnts[keypnt_mask][:, 1:]  # (y, z)
        src_keypnt_slice = src_keypnts[keypnt_mask][:, 1:]  # (y, z)
    elif select_dim == 1:
        keypnt_mask = np.logical_and(np.abs(tgt_keypnts[:, 1] - select_slice) <= 1,
                                     np.abs(src_keypnts[:, 1] - select_slice) <= 1)
        tgt_keypnt_slice = tgt_keypnts[keypnt_mask][:, [0, 2]]  # (x, z)
        src_keypnt_slice = src_keypnts[keypnt_mask][:, [0, 2]]  # (x, z)
    else:
        keypnt_mask = np.logical_and(np.abs(tgt_keypnts[:, 2] - select_slice) <= 1,
                                     np.abs(src_keypnts[:, 2] - select_slice) <= 1)
        tgt_keypnt_slice = tgt_keypnts[keypnt_mask][:, :2]  # (x, y)
        src_keypnt_slice = src_keypnts[keypnt_mask][:, :2]  # (x, y)

    if len(tgt_keypnt_slice) > 0:
        ax.scatter(tgt_keypnt_slice[:, 0], tgt_keypnt_slice[:, 1],
            c='#4878d0', marker='+', s=100,
            # label="Keypoints"
            )
        ax.scatter(src_keypnt_slice[:, 0], src_keypnt_slice[:, 1],
            c='#ee854a', marker='+', s=100,
            # label="Keypoints"
            )
        # Larger, more visible markers with edge
        # ax.scatter(tgt_keypnt_slice[:, 0], tgt_keypnt_slice[:, 1],
        #     c='lime', marker='o', s=50,
        #     edgecolors='black', linewidths=1.5,
        #     label="Target keypoints", zorder=10)

        # ax.scatter(src_keypnt_slice[:, 0], src_keypnt_slice[:, 1],
        #     c='magenta', marker='s', s=50,  # square marker to differentiate
        #     edgecolors='black', linewidths=1.5,
        #     label="Source keypoints", zorder=10)

    # Draw lines connecting corresponding keypoints
    # for i in range(len(tgt_keypnt_slice)):
    #     ax.plot([tgt_keypnt_slice[i, 0], src_keypnt_slice[i, 0]],
    #             [tgt_keypnt_slice[i, 1], src_keypnt_slice[i, 1]],
    #             'red', linestyle='--', linewidth=2, alpha=0.7, zorder=5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
