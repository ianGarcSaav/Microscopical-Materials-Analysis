from scipy import ndimage
from skimage import color
import cv2
import numpy as np

def label_components(mask):
    connectivity = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)
    return labeled_mask, num_labels

def color_clusters(labeled_mask, cluster_labels=None):
    """
    Colors the clusters in the labeled mask.

    Args:
        labeled_mask (numpy.ndarray): The labeled mask.
        cluster_labels (dict, optional): A dictionary mapping cluster labels to K-Means cluster assignments.
                                         If None, the clusters are colored randomly. Defaults to None.

    Returns:
        numpy.ndarray: The colored image.
    """
    if cluster_labels is None:
        # If no cluster labels are provided, color the clusters randomly
        return color.label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0))
    else:
        # Create a color map based on the K-Means cluster assignments
        unique_labels = np.unique(labeled_mask[labeled_mask != 0])
        
        # Generate a color for each K-Means cluster
        kmeans_clusters = set(cluster_labels.values())
        color_map = {cluster: np.random.rand(3) for cluster in kmeans_clusters}

        # Assign colors to each label based on its K-Means cluster assignment
        colored_img = np.zeros((labeled_mask.shape[0], labeled_mask.shape[1], 3))
        for label in unique_labels:
            kmeans_cluster = cluster_labels.get(label)
            if kmeans_cluster is not None:
                colored_img[labeled_mask == label] = color_map[kmeans_cluster]
            else:
                colored_img[labeled_mask == label] = [0,0,0]  # Black if no cluster assignment

        return colored_img

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
