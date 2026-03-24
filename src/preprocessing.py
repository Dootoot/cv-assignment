# leo's note
# median and bilateral filters are both good at preserving edges but smoothes out
# regions of similar 'appearance'

# an example of how we could use this is
# image -> apply filter -> superpixel clustering -> label superpixels in model
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries


# good for removing things like salt and pepper noise but may remove too much detail
# adjust sampling diameter as odd numbers as needed
def apply_median_filter(image: np.ndarray, sampling_diameter = 5) -> np.ndarray:

    if image is None:
        raise RuntimeError()
    
    median_filtered_image = cv2.medianBlur(image, sampling_diameter)

    return median_filtered_image

def apply_bilateral_filter(image: np.ndarray, sampling_diameter = 5, sigma_colour = 15, sigma_space = 15) -> np.ndarray:

    if image is None:
        raise RuntimeError()
    """
    bilateralFilter(image, d, sigma colour, sigma space)

    diameter = sampling range for the gaussian curve
    sigma colour = standard deviation for colour sampling curve
    sigma space = same as gaussian blur, linear component

    Generally want to keep diameter small (3, 5, or 7) to prevent overblurring and edge maintenance
    SD for colour and space needs to be adjusted in testing as 
    weight = exp(-differnece(intensity1, intensity2)^2 / 2*sigma^2))
    """
    bilateral_filtered_image = cv2.bilateralFilter(image, sampling_diameter, sigma_colour, sigma_space)

    return bilateral_filtered_image

#
def apply_SLIC_superpixel(image: np.ndarray, num_segments = 400, compactness = 15) -> np.ndarray:
    # currently assume image is in BGR (cv reads RGB as BGR)
    image_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    superpixel_image = slic(image_Lab, n_segments = num_segments, compactness = compactness, start_label = 1)

    return superpixel_image


# use kernel_size = -1 for Scharr's filter which is better for orientation preservation
def apply_sobel_filter(image: np.ndarray, kernel_size = 3, shifting_delta = 0) -> np.ndarray:

    if image is None:
        raise RuntimeError()

    # float64 to prevent negatives being set to 0
    x_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, k_size = kernel_size, delta = shifting_delta) # type: ignore
    y_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, k_size = kernel_size, delta = shifting_delta) # type: ignore

    # just as a note, .Sobel() returns a convolution of image array made of its directional derivatives as entries to the image matrix

    # standarised to use L2 norm, could consider L1 norm if we run into computation time issues but this shouldn't happen
    sobel_filtered_image = np.sqrt(x_sobel**2 + y_sobel**2)

    return sobel_filtered_image