# try watershed segmentation since crops are often close together and touching each other,
# so i would like to try watershed algorithm to separate them

import cv2
import numpy as np
from pathlib import Path

from src import preprocessing, feature_extraction

def watershed(image_folder_path: str):
    if image_folder_path is None:
        raise RuntimeError("Invalid image folder path")
    
    images_dir = Path(image_folder_path)

    for img in images_dir.glob("*.png"):
        if not img is None and not img.name.endswith("_mask.png"):
            result = watershed_pipeline_on_image(str(img))
            if result is None:
                continue

            save_comparision(result, 
            cv2.imread(str(img).removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE), 
            cv2.imread(str(img), cv2.IMREAD_GRAYSCALE), 
            "output/watershed_comparision", 
            img.name)


def watershed_pipeline_on_image(image_path: str):
    # read as bgr
    image = cv2.imread(image_path)

    if image is None:
        return

    # apply bilateral filter to smoth images while preserving edges
    bilateral_filtered_image = preprocessing.apply_bilateral_filter(image)

    # feature enhancement
    feature_enhanced_image = feature_extraction.extract_normalised_excess_green_index(bilateral_filtered_image)
    img_8u = cv2.normalize(feature_enhanced_image, None, 0, 255,
                       cv2.NORM_MINMAX).astype(np.uint8)

    # thresholding to get binary image for watershed
    ret, binary_img = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removel with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary_img_opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations = 5)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations = 5)

    # sure background area
    sure_bg = cv2.dilate(binary_img_opened, kernel, iterations=3)

    # Distance transform
    dist = cv2.distanceTransform(binary_img_opened, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    #foreground area
    ret, sure_fg = cv2.threshold(dist, 0.02 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # watershed algorithm
    watershed_result = cv2.watershed(image, markers)

    binary_mask = np.zeros_like(watershed_result, dtype=np.uint8)
    binary_mask[watershed_result > 1] = 255
    return 255 - binary_mask

def save_comparision(predicted_mask, ground_truth, original_image, output_dir, filename):
    ground_truth_img = None
    if len(ground_truth.shape) == 2:
        ground_truth_img = ground_truth
    else:
        # this is just for safety net, ground truth should already be in grayscale but if not we can convert it
        ground_truth_img = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    combined_img = np.hstack([predicted_mask, ground_truth_img, original_image])
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    cv2.imwrite(str(Path(output_dir) / filename), combined_img)


"""
it works for some images but not all, it dosent performs well when there are significant shadow and lighting variation or when soil make up the
majority of the image. still need to improve, maybe mainly in the pre-processing and feature enhancement stage.

- maybe combine with random forest, use random forest for foreground and backgroung segmentation and use watershed method for instance segmentation

"""