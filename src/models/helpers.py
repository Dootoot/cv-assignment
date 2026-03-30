import numpy as np
import cv2
from typing import Any
from skimage.measure import regionprops
from .. import preprocessing
from .. import feature_extraction

import joblib
from pathlib import Path
"""
Helper functions for loading and storing models, as well as their prediction output
"""

# save and load model at model_dir, filename just has to be reasonable
# call it whatever u want tbh, probably want to include some info about the model in the filename for sanity sake, e.g. "random_forest_model.joblib"
def save_model(model, model_dir, filename):
    path = Path(model_dir) / filename
    joblib.dump(model, path)

def load_model(model_dir, filename):
    path = Path(model_dir) / filename
    if not path.exists():
        return None

    return joblib.load(path)

def save_prediction_comparison(predicted_mask, ground_truth, output_dir, filename):
    ground_truth_img = None
    if len(ground_truth.shape) == 2:
        ground_truth_img = ground_truth
    else:
        # this is just for safety net, ground truth should already be in grayscale but if not we can convert it
        ground_truth_img = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    combined_img = np.hstack([predicted_mask, ground_truth_img])
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    cv2.imwrite(str(Path(output_dir) / filename), combined_img)

def predict_from_trained_model(output_dir: str, testing_folder: str, model, predict_function):
    if testing_folder is None:
        raise RuntimeError("Invalid image folder path")
    
    images_dir = Path(testing_folder)

    metrics_list = []

    for img in images_dir.glob("*.png"):
        if not img is None and not img.name.endswith("_mask.png"):

            predicted_mask = predict_function(model, str(img))
            if predicted_mask is None:
               continue
             
            ground_truth_image = cv2.imread(str(img).removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE)
            metrics = compute_metrics(predicted_mask, ground_truth_image)
            metrics_list.append(metrics)

            original_name = img.stem
            save_prediction_comparison(predicted_mask, ground_truth_image, output_dir, f"{original_name}_predicted_against_truth.png")
    
    # return metrics score by averaging over image
    precision_total = 0
    recall_total = 0
    f1_total = 0
    iou_total = 0
    for metric in metrics_list:
        precision_total += metric["precision"]
        recall_total += metric["recall"]
        f1_total += metric["f1_score"]
        iou_total += metric["intersection_over_union"]

    n = len(metrics_list)
    return {
        "precision": precision_total / n,
        "recall": recall_total / n,
        "f1_score": f1_total / n,
        "intersection_over_union": iou_total / n
    }
"""
Helper functions for random forest model, including feature extraction and metric computation.
Can be reused for xgboost model as well without much change, if at all.
"""
def construct_superpixel_feature_matrix(superpixel_labels: np.ndarray, colour_maps: list, lbp_map, sobel_map) -> np.ndarray:
    # lab_a, g, s, exG
    STD_INDEX = [1, 4, 7, 9]
    labels = np.unique(superpixel_labels)

    n_superpixels = len(labels)
    # 9 colour means, 4 colour stds, 1 exG mean, 10 lbp histogram bins, 2 sobel stats, 3 shape features
    n_features = 10 + 4 + 10 + 2 + 3 # 29 features in total

    # shape properties from superpixel labels
    # docs indicates its more efficient to just compute single time outside of loop
    shape_props = {p.label: p for p in regionprops(superpixel_labels)}
    output_matrix = np.empty((n_superpixels, n_features))

    # colour features handled first from columns 0 - 9 for means, 10 - 12 for std

    for index, label in enumerate(labels):
        mask = (superpixel_labels == label)

        colour_means = []
        for i in range(0, len(colour_maps)):
            specific_map = colour_maps[i]
            mean = np.mean(specific_map[mask])
            colour_means.append(mean)

        colour_stds = []
        for i in STD_INDEX:
            specific_map = colour_maps[i]
            std = np.std(specific_map[mask])
            colour_stds.append(std)
        
        lbp_pixels = lbp_map[mask]
        hist, _ = np.histogram(lbp_pixels, bins = 10, range = (0, 10))
        hist_normalised = hist / len(lbp_pixels)

        sobel_pixels = sobel_map[mask]
        sobel_mean = np.mean(sobel_pixels)
        sobel_std = np.std(sobel_pixels)

        # shape features: area, eccentricity, compactness (perimeter^2 / 4*pi*area)
        props = shape_props[label]
        area = props.area
        eccentricity = props.eccentricity
        perimeter = props.perimeter
        # e^-5 prevents divison by 0 issue as a safeguard
        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-5)

        output_matrix[index] = np.concatenate([colour_means, colour_stds, hist_normalised, [sobel_mean, sobel_std, area, eccentricity, compactness]])
    
    return output_matrix

# assume binary_mask is read into memory already
def superpixel_label_from_ground_truth(superpixel_labels: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    labels = np.unique(superpixel_labels)
    n_superpixels = len(labels)
    output_matrix = np.zeros(n_superpixels)


    for index, label in enumerate(labels):
        mask = (superpixel_labels == label)
        if np.mean(binary_mask[mask] > 0) > 0.5:
            output_matrix[index] = 1

    return output_matrix

def extract_feature_maps(image) -> dict[str, Any]:
    if image is None:
        raise RuntimeError("shouldn't be possible to reach here")
    
    median_filtered_image = preprocessing.apply_median_filter(image)
    bilateral_filtered_image = preprocessing.apply_bilateral_filter(image)

    # colour extraction from bilateral filtered image
    lab_l, lab_a, lab_b = feature_extraction.extract_lab_channels(bilateral_filtered_image)
    h, s, v = feature_extraction.extract_hsv_channels(bilateral_filtered_image)
    r, g, b = feature_extraction.extract_rgb_channels(bilateral_filtered_image)
    exG = feature_extraction.extract_normalised_excess_green_index(bilateral_filtered_image)

    # median removes high intensity noise but maintains edges well
    sobel_map = preprocessing.apply_sobel_filter(median_filtered_image)

    # bilateral filter may lose fine information but macro-scale texture is more important and keeps image source consistent
    # for super pixel region labelling (spatially aligned essentially)
    lbp_map = feature_extraction.extract_lbp_texture_map(bilateral_filtered_image)

    superpixel_labels = preprocessing.apply_SLIC_superpixel(bilateral_filtered_image)

    return {
        "colour_maps": [lab_l, lab_a, lab_b, h, s, v, r, g, b, exG],
        "lbp_map": lbp_map,
        "sobel_map": sobel_map,
        "superpixel_labels": superpixel_labels
    }

def compute_metrics (predicted_mask, ground_truth) -> dict[str, Any]:
    # black or white so we can just cast them as > 0 or = 0 
    # just skips conversion from 255 -> 1 if binary masks uses diff values
    predicted_hits = (predicted_mask > 0)
    ground_truth_hits = (ground_truth > 0)

    TP = np.sum(predicted_hits & ground_truth_hits)
    FP = np.sum(predicted_hits & ~ground_truth_hits)
    FN = np.sum(~predicted_hits & ground_truth_hits)

    # formulas from slide
    non_zero_guarantee = 1e-10
    precision = TP / (TP + FP + non_zero_guarantee)
    recall = TP / (TP + FN + non_zero_guarantee)
    f1_score = 2 * precision * recall / (precision + recall + non_zero_guarantee)
    intersection_over_union = TP / (TP + FP + FN + non_zero_guarantee)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "intersection_over_union": intersection_over_union
    }