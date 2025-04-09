from scipy import ndimage
from skimage import color
import cv2
import numpy as np

def label_components(mask):
    # Etiquetar componentes conectados usando cv2.connectedComponents
    num_labels, labeled_mask = cv2.connectedComponents(mask.astype(np.uint8))
    return labeled_mask, num_labels

def color_clusters(labeled_mask):
    # Ensure background label is set correctly
    return color.label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0))

def label_components_watershed(mask):
    # Convert mask to binary image (0,255)
    binary = (mask.astype(np.uint8) * 255)
    
    # Compute distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)
    markers[markers == -1] = 0  # Set watershed boundaries to 0
    
    return markers

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

def classify_shapes(measurements):
    classifications = {}
    for measurement in measurements:
        label = measurement[0]
        area = measurement[1]
        perimeter = measurement[5]
        major_axis = measurement[4]
        minor_axis = measurement[5]

        # Calcular circularidad y relación de aspecto
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0

        # Clasificar según circularidad y relación de aspecto
        if circularity > 0.8:
            classifications[label] = "circle"
        elif aspect_ratio > 1.2:
            classifications[label] = "oval"
        else:
            classifications[label] = "diamond"
    return classifications
