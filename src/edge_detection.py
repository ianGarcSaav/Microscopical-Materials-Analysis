import numpy as np
from scipy import ndimage
import imageio

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
    """(c) Thin edges by keeping only local maxima."""
    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = theta * 180. / np.pi
    angle[angle<0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255; r = 255
            # 0°
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            # 45°
            elif 22.5 <= angle[i,j] < 67.5:
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            # 90°
            elif 67.5 <= angle[i,j] < 112.5:
                q = G[i+1, j]
                r = G[i-1, j]
            # 135°
            elif 112.5 <= angle[i,j] < 157.5:
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0
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
    """(e) Track edges by hysteresis: keep weak if connected to strong."""
    M, N = img.shape
    # Iterate through image pixels
    for i in range(1, M-1):
        for j in range(1, N-1):
            # If the pixel is weak
            if img[i,j] == weak:
                # Check its 8 neighbors for a strong pixel
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    # If connected to strong, promote weak to strong
                    img[i,j] = strong
                else:
                    # If not connected to strong, suppress (set to 0)
                    img[i,j] = 0
    # Return the final edge map (only strong pixels remain)
    # Note: The input 'img' is modified in place and returned.
    # Final map will have 0s and 'strong' values (e.g., 255)
    return img

def canny_edge_detector(image: np.ndarray,
                        sigma: float = 1.4,
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

    img = imageio.imread(sys.argv[1], as_gray=True).astype(np.float32)
    # Example of calling with specific parameters if needed:
    # edge_map = canny_edge_detector(img, sigma=1.0, low_ratio=0.08, high_ratio=0.18)
    edge_map = canny_edge_detector(img) # Uses the new defaults

    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.axis('off')
    plt.subplot(122), plt.imshow(edge_map, cmap='gray'), plt.title('Canny Edges')
    plt.axis('off')
    plt.show()