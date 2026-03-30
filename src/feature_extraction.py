# we need a few of each type to experiment and test with, probably at least one for each of the following

# colour (RGB, HSV, Lab -> 3-tuples of colour spaces -> 9 features)

# texture (maybe SIFT, or gray-level co-occurence matrices for stem contrast with leaf as a pattern etc.)
# superpixel shape (centroid, wheat heads are more rectangular/oval than stems etc.)

# example pipeline might be median filter + bilateral filter -> superpixel -> sobel filter
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def nothing():
    pass

def extract_lbp_texture_map(image_bgr):
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # uniform for rotation invariance, useful for analysing rotated images
    lbp = local_binary_pattern(image_gray, P = 8, R = 1, method = "uniform")

    return lbp

"""
Miscallaneous colour extraction helpers, returns a triplet
"""
# cv2 reads image as bgr instead of rgb by default
def extract_lab_channels(image_bgr):
    # convert and split
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    return l, a, b

def extract_hsv_channels(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)

    return h, s, v

def extract_rgb_channels(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)

    return r, g, b

def extract_excess_green_index(image_bgr):
    r, g, b = extract_rgb_channels(image_bgr)
    # might overflow so just typecasted to float32 to be a bit safer
    ee_index = 2 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)

    return ee_index

# normalised might be better since it is robust against lighting variation
def extract_normalised_excess_green_index(image_bgr):
    r, g, b = extract_rgb_channels(image_bgr)
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)

    # added e^-5 (very small number) to prevent division by 0 issue
    return (2 * g - r - b) / (r + g + b + 1e-5)