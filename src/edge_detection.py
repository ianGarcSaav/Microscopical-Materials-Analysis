import numpy as np
from scipy import ndimage


def gaussian_filter(image: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    """(a) Smooth the image with a Gaussian filter."""
    return ndimage.gaussian_filter(image, sigma)

def sobel_filters(image: np.ndarray):
    """(b) Compute gradient magnitude and direction via Sobel operators."""
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)
    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def non_max_suppression(G: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(c) Thin edges by keeping only local maxima using vectorized operations."""
    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.float32)
    
    # Convert angles to degrees and shift to positive values
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
    # Pad the gradient magnitude array
    G_pad = np.pad(G, pad_width=1, mode='constant', constant_values=0)
    
    # Create arrays for the 8 neighboring pixels
    n1 = G_pad[1:-1, 2:]  # right
    n2 = G_pad[2:, 2:]    # bottom-right
    n3 = G_pad[2:, 1:-1]  # bottom
    n4 = G_pad[2:, :-2]   # bottom-left
    n5 = G_pad[1:-1, :-2] # left
    n6 = G_pad[:-2, :-2]  # top-left
    n7 = G_pad[:-2, 1:-1] # top
    n8 = G_pad[:-2, 2:]   # top-right
    
    # Create masks for different angle ran  ges
    mask1 = ((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))
    mask2 = (22.5 <= angle) & (angle < 67.5)
    mask3 = (67.5 <= angle) & (angle < 112.5)
    mask4 = (112.5 <= angle) & (angle < 157.5)
    
    # Apply non-maximum suppression based on angle masks
    Z[mask1] = ((G[mask1] >= n1[mask1]) & (G[mask1] >= n5[mask1])) * G[mask1]
    Z[mask2] = ((G[mask2] >= n2[mask2]) & (G[mask2] >= n6[mask2])) * G[mask2]
    Z[mask3] = ((G[mask3] >= n3[mask3]) & (G[mask3] >= n7[mask3])) * G[mask3]
    Z[mask4] = ((G[mask4] >= n4[mask4]) & (G[mask4] >= n8[mask4])) * G[mask4]
    
    return Z

def double_threshold(img: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15):
    """(d) Classify pixels into strong, weak and non‑edges."""
    high = img.max() * high_ratio
    # Note: low threshold is calculated relative to high threshold
    low = high * low_ratio
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    # Define fixed values for weak and strong pixels in the result map
    strong = np.int32(255)
    weak = np.int32(75) # Adjusted weak value slightly for better distinction if needed, can be kept lower

    strong_i, strong_j = np.where(img >= high)
    # Pixels between low and high are weak
    weak_i, weak_j = np.where((img < high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res, weak, strong) # Return the map and the values used for weak/strong

def hysteresis(img: np.ndarray, weak: int, strong: int = 255) -> np.ndarray:
    """(e) Track edges by hysteresis using more efficient operations."""
    M, N = img.shape
    # Create a padded version of the image for neighbor operations
    padded = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    
    # Find weak pixels
    weak_pixels = img == weak
    
    while True:
        # Find weak pixels that have at least one strong neighbor
        neighbors = np.zeros_like(weak_pixels, dtype=bool)
        
        # Check all 8 neighbors efficiently using padded array
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                # Use correct slice of padded array matching original dimensions
                neighbors |= (padded[i:i+M, j:j+N] == strong)
        
        # Update weak pixels that should be promoted
        promote = weak_pixels & neighbors
        if not promote.any():
            break
            
        img[promote] = strong
        weak_pixels[promote] = False
    
    # Set remaining weak pixels to 0
    img[img == weak] = 0
    return img

def canny_edge_detector(image: np.ndarray,
                        sigma: float = 1.2,
                        # Increased default low_ratio for stricter weak pixel threshold
                        low_ratio: float = 0.10, # <-- Increased from 0.05
                        high_ratio: float = 0.15) -> np.ndarray:
    """Full pipeline: (a)–(e).
    Note: Default low_ratio increased for potentially cleaner edges.
    Optimal values depend on the image and should be tuned when calling.
    """
    smooth = gaussian_filter(image, sigma)
    G, theta = sobel_filters(smooth)
    non_max = non_max_suppression(G, theta)
    # Pass the adjusted ratios to double_threshold
    dt, weak_val, strong_val = double_threshold(non_max, low_ratio, high_ratio)
    # Pass the specific weak/strong values used by double_threshold to hysteresis
    edges = hysteresis(dt, weak_val, strong_val)
    # Ensure the final output is uint8 with 0 and 255 values
    edges_uint8 = (edges == strong_val).astype(np.uint8) * 255
    return edges_uint8

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python edge_detection.py path/to/image")
        sys.exit(1)
