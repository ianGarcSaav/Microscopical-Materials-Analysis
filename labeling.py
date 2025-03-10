from scipy import ndimage
from skimage import color

def label_components(mask):
    connectivity = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)
    return labeled_mask, num_labels

def color_clusters(labeled_mask):
    return color.label2rgb(labeled_mask, bg_label=0)
