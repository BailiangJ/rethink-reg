import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import map_coordinates
import matplotlib.colors as mcolors
from scipy.interpolate import BSpline, splrep

DEFAULT_LUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'FreeSurferColorLUT.txt')


def read_freesurfer_lut(file_path:Optional[str]):
    if file_path is None:
        file_path = DEFAULT_LUT
    lut = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                label = int(parts[0])
                color = tuple(map(int, parts[2:6]))
                lut[label] = color
    return lut


def apply_lut(seg, lut):
    color_image = np.zeros(seg.shape + (4,), dtype=np.uint8)  # RGBA
    unique_labels = np.unique(seg)
    for label in unique_labels:
        if label in lut:
            color_image[seg == label] = lut[label]
    return color_image

def plot_image_slice(
    img:np.ndarray,
    select_dim:int,
    select_slice:int,
    save_path:str,
    figsize: tuple=(10,8),
):
    img = np.take(img, indices=select_slice, axis=select_dim)
    img = (img - img.min()) / (img.max() - img.min())
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_seg_slice(
    seg:np.ndarray,
    select_dim:int,
    select_slice:int,
    save_path:str,
    figsize:tuple=(10,8),
):
    seg = np.take(seg, indices=select_slice, axis=select_dim)  # label map
    freesurfer_lut = read_freesurfer_lut(DEFAULT_LUT)
    seg_rgba = apply_lut(seg, freesurfer_lut)[...,:3]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(seg_rgba)
    ax.axis('off')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_grid(x, y,
              save_name: str,
              figsize=(10, 8),
              rgb=None,
              **kwargs):
    figsize = (figsize[1], figsize[0])

    fig, ax = plt.subplots(figsize=figsize)

    # print(x.shape, x.T.shape)
    x = x.T
    y = y.T
    segs1 = np.stack((y, -x), axis=2)
    segs2 = segs1.transpose(1, 0, 2)

    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.axis('off')
    ax.autoscale()
    # fig.savefig(save_name, dpi=300, bbox_inches='tight')
    return fig,ax

def plot_deformed_grid(disp: np.ndarray,
                       select_dim: int,
                       select_slice: int,
                       # img_shape: tuple,
                       step_size: int,
                       save_path: str,
                       figsize=(10,8)):
    '''
    Args:
        disp: displacement field in image space (not normalized to [-1, 1]), shape (H, W, D, 3)
        select_dim: selected dimension.
        select_slice: selected slice.
        step_size:
        save_path:
        figsize:

    Returns:

    '''
    dims = [0, 1, 2]
    dims.pop(select_dim)
    fig_height = figsize[1]
    height, width = disp.shape[dims[0]], disp.shape[dims[1]]
    aspect_ratio = width / height
    print(aspect_ratio, fig_height*width/height)
    fig_width = fig_height * aspect_ratio
    figsize=(fig_height, fig_width)

    select_slice = select_slice // step_size
    # set indexing='ij' whenever using np.meshgrid
    img_shape = disp.shape[:3]
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(0, img_shape[0] + 1, step_size),
        np.arange(0, img_shape[1] + 1, step_size),
        np.arange(0, img_shape[2] + 1, step_size),
        indexing='ij')

    id_grids = [grid_x, grid_y, grid_z]

    disp_x = map_coordinates(disp[..., 0], (grid_x, grid_y, grid_z), order=0)
    disp_y = map_coordinates(disp[..., 1], (grid_x, grid_y, grid_z), order=0)
    disp_z = map_coordinates(disp[..., 2], (grid_x, grid_y, grid_z), order=0)
    disp_fields = [disp_x, disp_y, disp_z]

    def_grid = []
    for i in dims:
        def_grid.append(np.take(id_grids[i],
                                indices=select_slice,
                                axis=select_dim) + np.take(disp_fields[i],
                                                           indices=select_slice,
                                                           axis=select_dim))

    fig, ax = plot_grid(def_grid[0], def_grid[1], rgb=None, save_name=save_path,
                        figsize=figsize,
                        color="red")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hist(x:np.ndarray, y:np.ndarray, save_path:str, figsize=(10,8),
              label_fontsize: Optional[int]=24,
              tick_fontsize: Optional[int]=20,
              legend_fontsize: Optional[int]=24):
    plt.figure(figsize=figsize)
    plt.hist(x.flatten(), bins=50, alpha=0.7, color='blue', density=True, label='original')
    plt.hist(y.flatten(), bins=50, alpha=0.7, color='red', density=True, label='augmented')
    if label_fontsize is not None:
        plt.xlabel('Intensity', fontsize=label_fontsize)
        plt.ylabel('Density', fontsize=label_fontsize)
    else:
        plt.xlabel('Intensity')
        plt.ylabel('Density')
    if legend_fontsize is not None:
        plt.legend(fontsize=legend_fontsize)
    else:
        plt.legend()
    if tick_fontsize is not None:
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_spline(spline:BSpline, save_path:str,
              label_fontsize: Optional[int]=18,
              tick_fontsize: Optional[int]=12):
    # Determine valid domain from knot vector
    k = spline.k
    t = np.asarray(spline.t)
    x_min = t[k]
    x_max = t[-k-1]
    print(x_min, x_max)

    # Sample spline densely over its domain
    x_eval = np.linspace(x_min, x_max, 1000)
    y_eval = spline(x_eval)

    # Normalize x to [0, 1] for plotting
    denom = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
    x_plot = (x_eval - x_min) / denom

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x_plot, y_eval, color='steelblue', linewidth=2.0, label='B-spline')

    # Extract unique knots within domain and plot them on the curve
    # knots_in_domain = t[(t >= x_min) & (t <= x_max)]
    # if knots_in_domain.size > 0:
    #     # Preserve order while making unique
    #     unique_idx = np.unique(knots_in_domain, return_index=True)[1]
    #     knots_unique = knots_in_domain[np.sort(unique_idx)]
    #     xk_plot = (knots_unique - x_min) / denom
    #     yk = spline(knots_unique)
    #     ax.scatter(xk_plot, yk, color='crimson', s=30, zorder=3, label='knots')
    #     for xv in xk_plot:
    #         ax.axvline(x=xv, color='crimson', alpha=0.15, linewidth=1.0)

    # Axes limits (both normalized to [0,1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # Labels (optional)
    ax.set_xlabel('original', fontsize=label_fontsize)
    ax.set_ylabel('augmented', fontsize=label_fontsize)

    # Ticks at fixed normalized locations
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Clean look: hide spines
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)

    # Draw axes with arrows along x=0..1 and y crossing at 0
    arrow_style = dict(arrowstyle='-|>', color='black', lw=1.5, shrinkA=0, shrinkB=0)
    # x-axis (arrow to the right). Anchor slightly beyond [0,1] for visibility
    ax.annotate('', xy=(1.03, 0.0), xytext=(-0.03, 0.0), xycoords=('data', 'data'),
                textcoords=('data', 'data'), arrowprops=arrow_style)
    # y-axis (arrow to the top). Use current ylim for head
    ylim = ax.get_ylim()
    ax.annotate('', xy=(0.0, ylim[1]), xytext=(0.0, ylim[0]),
                xycoords=('data', 'data'), textcoords=('data', 'data'), arrowprops=arrow_style)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_gaussian(save_path:str):
    x = np.linspace(-4.0, 4.0, 1000)
    y = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Make background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.plot(x, y, color='blue', linewidth=2.0)

    # Limits
    ax.set_xlim(-4.0, 4.0)
    y_max = float(y.max())
    ax.set_ylim(0.0, y_max * 1.1)

    # Hide ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)

    # Draw axes with arrows
    arrow_style = dict(arrowstyle='-|>', color='black', lw=1.5, shrinkA=0, shrinkB=0)
    # x-axis (arrow to the right)
    ax.annotate('', xy=(4.0, 0.0), xytext=(-4.0, 0.0), arrowprops=arrow_style)
    # y-axis (arrow to the top)
    ax.annotate('', xy=(0.0, y_max * 1.1), xytext=(0.0, 0.0), arrowprops=arrow_style)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def plot_gamma_contrast(gamma:float=1.0, invert:bool=False, save_path:str=None):
    # Domain [0,1]
    x = np.linspace(0.0, 1.0, 1000)
    x_in = 1.0 - x if invert else x
    y = np.power(x_in, gamma)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, color='steelblue', linewidth=2.0)

    # Axes limits and ticks
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel('original')
    ax.set_ylabel('augmented')

    # Hide spines
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)

    # Arrowed axes
    arrow_style = dict(arrowstyle='-|>', color='black', lw=1.5, shrinkA=0, shrinkB=0)
    ax.annotate('', xy=(1.03, 0.0), xytext=(-0.03, 0.0), xycoords=('data', 'data'),
                textcoords=('data', 'data'), arrowprops=arrow_style)
    ylim = ax.get_ylim()
    ax.annotate('', xy=(0.0, ylim[1]), xytext=(0.0, ylim[0]),
                xycoords=('data', 'data'), textcoords=('data', 'data'), arrowprops=arrow_style)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def norm_img(x):
    return (x - x.min()) / (x.max() - x.min())
