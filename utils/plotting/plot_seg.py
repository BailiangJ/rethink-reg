import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from typing import Optional, Sequence
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

def plot_seg(
    image:np.ndarray,
    label_seg:Optional[np.ndarray]=None,
    contour_seg:Optional[np.ndarray]=None,
    label_alpha:float=0.35,
    label_cmap:str="spring",
    contour_alpha:float=1.0,
    contour_cmap:Optional[str]=None,
    contour_linewidth:float=2,
    save_path:str="img_w_seg.png",
    figsize:Sequence[int]=(10,8),
):
    """
    Visualize image with label segmentation A and/or label contour B.
    A and B are different segmentation. Either one can be None.

    Args:
        image: np.ndarray, of shape (H, W) / (H, W, 3)
        label_seg: Optional[np.ndarray], segmentation A
                   of shape (H, W), label segmentation
        contour_seg: Optional[np.ndarray], segmentation B
                     of shape (H, W), label segmentation
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Check if image is grayscale or RGB and display
    if len(image.shape) == 2:
        # Grayscale image
        ax.imshow(image, cmap='gray')
    elif len(image.shape) == 3 and image.shape[-1] == 3:
        # RGB image
        ax.imshow(image)
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")


    # Plot label segmentation A (only if provided)
    if label_seg is not None:
        masked_label_seg = np.ma.masked_where(label_seg==0, label_seg)
        ax.imshow(masked_label_seg, cmap=label_cmap, alpha=label_alpha, interpolation="none") # no interpolation for cmap

    if contour_seg is not None:
        # Determine which classes to plot contours for
        if label_seg is not None:
            # Check if segmentation A and B share same classes
            label_classes = np.unique(label_seg)
            contour_classes = np.unique(contour_seg)
            shared_classes = np.intersect1d(label_classes, contour_classes)

            # Remove background class (0) if present
            shared_classes = shared_classes[shared_classes != 0]
        else:
            # If label_seg is None, plot contours for all classes in contour_seg
            contour_classes = np.unique(contour_seg)
            shared_classes = contour_classes[contour_classes != 0]  # Remove background class

        # Plot contour segmentation B
        # Get colormap for contours
        if contour_cmap is None:
            # Use same colormap as labels but create a modified version
            contour_cmap = modify_colormap(label_cmap, brightness_factor=1.2, saturation_factor=1.2)
        else:
            contour_cmap = plt.get_cmap(contour_cmap)

        # Find contours for each class separately
        for class_id in shared_classes:
            # Create binary mask for current class
            class_mask = (contour_seg == class_id).astype(np.uint8)

            # Find contours using skimage.measure.find_contours
            contours = measure.find_contours(class_mask, level=0.5)

            # Get color for this class from colormap
            # Normalize class_id to [0, 1] range for colormap
            if len(shared_classes) > 1:
                norm_value = (class_id - shared_classes.min()) / (shared_classes.max() - shared_classes.min())
            else:
                norm_value = 0.5

            color = contour_cmap(norm_value)

            # Plot all contours for this class
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0],
                    color=color, alpha=contour_alpha,
                    linewidth=contour_linewidth)

    ax.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def modify_colormap(base_cmap_name: str,
                            brightness_factor: float = 0.8,
                           saturation_factor: float = 1.2):
    """
    Create a modified colormap with adjusted brightness/saturation.

    Args:
        base_cmap_name: Name of the base colormap
        brightness_factor: Factor to adjust brightness (< 1 = darker, > 1 = brighter)
        saturation_factor: Factor to adjust saturation (< 1 = less saturated, > 1 = more saturated)

    Returns:
        matplotlib colormap
    """
    base_cmap = plt.get_cmap(base_cmap_name)

    # Get colors from base colormap
    colors_rgba = base_cmap(np.linspace(0, 1, 256))

    # Convert to HSV for easier manipulation
    modified_colors = []
    for color in colors_rgba:
        r, g, b, a = color
        h, s, v = mcolors.rgb_to_hsv([r, g, b])

        # Modify saturation and brightness
        s = np.clip(s * saturation_factor, 0, 1)
        v = np.clip(v * brightness_factor, 0, 1)

        # Convert back to RGB
        r, g, b = mcolors.hsv_to_rgb([h, s, v])
        modified_colors.append([r, g, b, a])

    return ListedColormap(modified_colors)
