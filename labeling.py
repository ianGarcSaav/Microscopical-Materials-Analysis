import numpy as np
from scipy import ndimage

def label_clusters(mask):
    connectivity = np.ones((3, 3))
    labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)
    return labeled_mask, num_labels