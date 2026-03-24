# define our different ML models here
# leo's note, im gonna be trying random forests first
# other things could be interesting too

# we should probably compare deep learning methods to more traditiona ML pipelines in our write up
import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import preprocessing
import feature_extraction

def region_based_randomforest(image_folder_path: str):
    if image_folder_path is None:
        raise RuntimeError("Invalid image folder path")
    
    images_dir = Path(image_folder_path)

    X_all = []
    y_all = []

    for img in images_dir.glob("*.png"):

        if not img is None and not img.name.endswith("_mask.png"):
            result = region_based_randomforest_pipeline_on_image(str(img))
            if result is None:
                continue
            f, t = result

            X_all.append(f)
            y_all.append(t)
    
    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)

    # create forest object
    random_forest = RandomForestClassifier(n_estimators = 100, class_weight = "balanced", min_samples_leaf = 5, n_jobs = -1, random_state = 2006)
    random_forest.fit(X_train, y_train)

    return random_forest
    


def region_based_randomforest_pipeline_on_image(image_path: str):
    # read as bgr
    image = cv2.imread(image_path)
    binary_mask = cv2.imread(image_path.removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE)

    if image is None or binary_mask is None:
        return

    # preprocessing
    median_filtered_image = preprocessing.apply_median_filter(image)
    bilateral_filtered_image = preprocessing.apply_bilateral_filter(image)

    # colour extraction from bilateral filtered image
    lab_l, lab_a, lab_b = feature_extraction.extract_lab_channels(bilateral_filtered_image)
    h, s, v = feature_extraction.extract_hsv_channels(bilateral_filtered_image)
    r, g, b = feature_extraction.extract_rgb_channels(bilateral_filtered_image)
    exG = feature_extraction.extract_excess_green_index(bilateral_filtered_image)

    # median removes high intensity noise but maintains edges well
    sobel_map = preprocessing.apply_sobel_filter(median_filtered_image)

    # bilateral filter may lose fine information but macro-scale texture is more important and keeps image source consistent
    # for super pixel region labelling (spatially aligned essentially)
    lbp_map = feature_extraction.extract_lbp_texture_map(bilateral_filtered_image)

    superpixel_labels = preprocessing.apply_SLIC_superpixel(bilateral_filtered_image)

    feature_matrix = construct_superpixel_feature_matrix(superpixel_labels, [lab_l, lab_a, lab_b, h, s, v, r, g, b, exG], lbp_map, sobel_map)
    truth_map = super_pixel_label_from_ground_truth(superpixel_labels, binary_mask)

    return feature_matrix, truth_map


def construct_superpixel_feature_matrix(superpixel_labels: np.ndarray, feature_maps: list, lbp_map, sobel_map) -> np.ndarray:
    STD_INDEX = [1, 7, 9]
    labels = np.unique(superpixel_labels)

    n_superpixels = len(labels)
    # 9 coloour means, 3 colour stds, 1 exG mean, 10 lbp histogram bins, 2 sobel stats 
    n_features = 10 + 3 + 10 + 2 # 25 in total
    output_matrix = np.empty((n_superpixels, n_features))

    # colour features handled first from columns 0 - 9 for means, 10 - 12 for std

    for index, label in enumerate(labels):
        mask = (superpixel_labels == label)

        colour_means = []
        for i in range(0, len(feature_maps)):
            specific_map = feature_maps[i]
            mean = np.mean(specific_map[mask])
            colour_means.append(mean)

        colour_stds = []
        for i in STD_INDEX:
            specific_map = feature_maps[i]
            std = np.std(specific_map[mask])
            colour_stds.append(std)
        
        lbp_pixels = lbp_map[mask]
        hist, _ = np.histogram(lbp_pixels, bins = 10, range = (0, 10))
        hist_normalised = hist / len(lbp_pixels)

        sobel_pixels = sobel_map[mask]
        sobel_mean = np.mean(sobel_pixels)
        sobel_std = np.std(sobel_pixels)

        output_matrix[index] = np.concatenate([colour_means, colour_stds, hist_normalised, [sobel_mean, sobel_std]])
    
    return output_matrix

# assume binary_mask is read into memory already
def super_pixel_label_from_ground_truth(superpixel_labels: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    labels = np.unique(superpixel_labels)
    n_superpixels = len(labels)
    output_matrix = np.zeros(n_superpixels)


    for index, label in enumerate(labels):
        mask = (superpixel_labels == label)
        if np.mean(binary_mask[mask] > 0) > 0.5:
            output_matrix[index] = 1

    return output_matrix