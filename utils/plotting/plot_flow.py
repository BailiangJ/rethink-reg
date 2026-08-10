import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from matplotlib.pyplot import Axes, Figure

from typing import Optional,Sequence
from .flow_rgb import flow_to_color

def plot_deformed_grid(
    flow:np.ndarray,
    step_size:int,
    color:str='red',
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    figsize:Optional[Sequence[int]]=(10,8),
):
    '''
    Args:
        flow: 2D displacement field, of shape (H, W, 2)
        step_size: grid line step size
    '''
    H, W = flow.shape[:2]

    if ax is None and fig is None:
        aspect_ratio = W / H
        fig_height, fig_width = figsize
        fig_height = fig_width * aspect_ratio
        figsize = (fig_height, fig_width)
        fig, ax = plt.subplots(1,1,figsize=figsize)

    grid_y, grid_x = np.meshgrid(
                        np.arange(0, H, step_size),
                        np.arange(0, W, step_size),
                        indexing='ij'
                    )

    disp_y = flow[grid_y, grid_x, 0]
    disp_x = flow[grid_y, grid_x, 1]

    def_grid_y = grid_y + disp_y
    def_grid_x = grid_x + disp_x

    # Plot horizontal lines (constant y)
    for i in range(def_grid_y.shape[0]):
                    ax.plot(def_grid_x[i, :], def_grid_y[i, :], color, alpha=0.7, linewidth=0.8)

    # Plot vertical lines (constant x)
    for j in range(def_grid_y.shape[1]):
                    ax.plot(def_grid_x[:, j], def_grid_y[:, j], color, alpha=0.7, linewidth=0.8)
    return fig, ax

def quiver_plot(
    flow:np.ndarray,
    step_size:int,
    alpha:float=0.8,
    scale:float=0.75,
    width:float=0.0025,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    figsize:Optional[Sequence[int]]=(10,8),
):
    H, W = flow.shape[:2]

    if ax is None and fig is None:
        aspect_ratio = W / H
        fig_height, fig_width = figsize
        fig_height = fig_width * aspect_ratio
        figsize = (fig_height, fig_width)
        fig, ax = plt.subplots(1,1,figsize=figsize)

    grid_y, grid_x = np.meshgrid(
                        np.arange(0, H, step_size),
                        np.arange(0, W, step_size),
                        indexing='ij'
                    )

    u_quiver = flow[::step_size, ::step_size, 1]
    v_quiver = flow[::step_size, ::step_size, 0]

    ax.quiver(grid_x, grid_y, u_quiver, v_quiver,
            color='black',
            alpha=alpha,
            scale=scale,
            width=width,
            angles='xy',
            scale_units='xy'
            )
    return fig, ax

def plot_flow_rgb(
    flow:np.ndarray,
    fig:Optional[Figure]=None,
    ax:Optional[Axes]=None,
    figsize:Optional[Sequence[int]]=(10,8),
):
    H, W = flow.shape[:2]

    if ax is None and fig is None:
        aspect_ratio = W / H
        fig_height, fig_width = figsize
        fig_height = fig_width * aspect_ratio
        figsize = (fig_height, fig_width)
        fig, ax = plt.subplots(1,1,figsize=figsize)

    flow_img = flow_to_color(flow, convert_to_bgr=False)
    ax.imshow(flow_img)
    return fig, ax
