from scipy import ndimage
from skimage import color
import cv2
import numpy as np

def label_components(mask):
    connectivity = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)
    return labeled_mask, num_labels

def color_clusters(labeled_mask):
    # Ensure background label is set correctly
    return color.label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0))

def label_components_watershed(mask):
    # Convert mask to binary image (0,255)
    binary = (mask.astype(np.uint8) * 255)
    # Apply Gaussian blur to reduce noise while preserving edges
    binary_blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    # Compute distance transform
    dist = cv2.distanceTransform(binary_blurred, cv2.DIST_L2, 5)
    # Threshold to obtain sure foreground regions
    ret, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Identify unknown regions
    unknown = cv2.subtract(binary_blurred, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # ensure background is not labelled as 0
    markers[unknown == 255] = 0
    # Apply watershed segmentation (requires a 3-channel image)
    markers = cv2.watershed(cv2.cvtColor(binary_blurred, cv2.COLOR_GRAY2BGR), markers)
    # Set boundaries (-1) to 0
    markers[markers == -1] = 0
    labeled_mask = markers
    num_labels = np.max(markers)
    return labeled_mask, num_labels

def label_components_watershed_advanced(mask):
    # Convert mask to binary image (0,255)
    binary = (mask.astype(np.uint8) * 255)
    # Apply Gaussian blur to reduce noise
    binary_blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    # Compute morphological gradient to emphasize edges
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(binary_blurred, cv2.MORPH_GRADIENT, kernel)
    # Compute distance transform on blurred binary image
    dist = cv2.distanceTransform(binary_blurred, cv2.DIST_L2, 5)
    # Threshold to obtain sure foreground regions (adjust factor as needed)
    ret, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Dilate the blurred binary to get sure background regions
    sure_bg = cv2.dilate(binary_blurred, kernel, iterations=3)
    # Identify unknown regions
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # ensure background is not 0
    markers[unknown == 255] = 0
    # Apply watershed segmentation on the gradient image (converted to 3-channel)
    markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)
    # Set watershed boundaries (-1) to 0 for clarity
    markers[markers == -1] = 0
    labeled_mask = markers
    num_labels = np.max(markers)
    return labeled_mask, num_labels
