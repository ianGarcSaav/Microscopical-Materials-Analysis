from scipy import ndimage
from skimage import color
import numpy as np

def label_components(mask):
    """Labels connected components in a binary or integer mask."""
    # Define connectivity (8-connectivity in this case)
    connectivity = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])
    # Use ndimage.label. It treats any non-zero pixel as foreground.
    labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)
    # num_labels includes the background, so the number of actual components is num_labels
    return labeled_mask, num_labels

def color_clusters(labeled_mask):
    """Assigns colors to labeled regions for visualization."""
    # Use label2rgb to color the labels.
    # bg_label=0 ensures the background (label 0) is colored specifically (black by default).
    # bg_color=(0, 0, 0) explicitly sets background to black.
    # Returns an RGB image (float format, scale 0-1 usually).
    # Need to convert to uint8 later if saving with libraries like OpenCV.
    return color.label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0))