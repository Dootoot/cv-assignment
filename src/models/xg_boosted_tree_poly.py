import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from .helpers import (
    save_model, load_model,
    extract_feature_maps, construct_superpixel_feature_matrix,
    superpixel_label_from_ground_truth, predict_from_trained_model,
)

# LBP bins sum to exactly 1.0 per superpixel, making bin 9 linearly dependent
# on bins 0-8. Dropping it before expansion removes degenerate cross-terms.
# Feature layout: [0-9 colour means | 10-13 colour stds | 14-23 LBP bins | 24-25 Sobel | 26-28 shape]
_LBP_REDUNDANT_BIN_INDEX = 23


def build_poly_dataset(image_folder_path: str):
    images_dir = Path(image_folder_path)
    X_all, y_all, image_row_counts = [], [], []

    for img in sorted(images_dir.glob("*.png")):
        if img.name.endswith("_mask.png"):
            continue
        image = cv2.imread(str(img))
        binary_mask = cv2.imread(str(img).removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE)
        if image is None or binary_mask is None:
            continue
        maps = extract_feature_maps(image)
        X = construct_superpixel_feature_matrix(
            maps["superpixel_labels"], maps["colour_maps"], maps["lbp_map"], maps["sobel_map"]
        )
        y = superpixel_label_from_ground_truth(maps["superpixel_labels"], binary_mask)
        X_all.append(X)
        y_all.append(y)
        # track superpixel count per image for image-level CV splitting
        image_row_counts.append(len(y))

    X_raw = np.vstack(X_all)
    y_raw = np.concatenate(y_all).astype(int)
    X_raw = np.delete(X_raw, _LBP_REDUNDANT_BIN_INDEX, axis=1)
    return X_raw, y_raw, image_row_counts


def fit_scaler_and_poly(X_raw: np.ndarray):
    # scaler must be fit on training data only; use .transform() everywhere else
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    return scaler, poly, X_poly


def apply_cheap_filters(X_poly: np.ndarray, var_threshold: float = 0.01, corr_threshold: float = 0.95):
    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X_poly)
    surviving = np.where(vt.get_support())[0]

    # greedy forward pass: drop feature j if |ρ(i,j)| > threshold for any earlier i
    X_sub = X_poly[:, surviving]
    corr = np.corrcoef(X_sub.T)
    drop_local = set()
    p = corr.shape[0]
    for i in range(p):
        if i in drop_local:
            continue
        for j in range(i + 1, p):
            if j not in drop_local and abs(corr[i, j]) > corr_threshold:
                drop_local.add(j)

    kept_local = np.array([i for i in range(p) if i not in drop_local])
    return surviving[kept_local]


def _fit_xgb_with_l1(X: np.ndarray, y: np.ndarray, reg_alpha: float = 1.0, colsample_bytree: float = 0.4):
    scale_pos_weight = float(np.sum(y == 0)) / float(np.sum(y == 1))
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=reg_alpha,
        reg_lambda=1.0,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=0.8,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=5,
        n_jobs=-1,
        random_state=2006,
        eval_metric="logloss",
    )
    xgb.fit(X, y)
    return xgb


def _gain_trim(xgb_model: XGBClassifier, current_indices: np.ndarray, keep_n: int):
    # rank by gain importance (reduction in training loss per split)
    importances = xgb_model.feature_importances_
    ranked = np.argsort(importances)[::-1]
    keep_local = np.sort(ranked[:keep_n])
    return current_indices[keep_local]


def iterative_prune(X_poly: np.ndarray, y: np.ndarray, initial_indices: np.ndarray,
                    n_iterations: int = 4, drop_fraction: float = 0.25, min_features: int = 25):
    current_indices = initial_indices.copy()
    checkpoints = []

    for iteration in range(n_iterations):
        X_sub = X_poly[:, current_indices]
        xgb = _fit_xgb_with_l1(X_sub, y, reg_alpha=0.5, colsample_bytree=0.5)

        n_keep = max(int(len(current_indices) * (1.0 - drop_fraction)), min_features)
        current_indices = _gain_trim(xgb, current_indices, n_keep)
        checkpoints.append({'iteration': iteration, 'n_features': len(current_indices), 'indices': current_indices.copy()})
        print(f"  Prune iter {iteration + 1}/{n_iterations}: {len(current_indices)} features remaining")

    return checkpoints


def _rows_for_images(image_indices, image_row_counts):
    row_starts = np.concatenate([[0], np.cumsum(image_row_counts)]).astype(int)
    row_lists = [np.arange(int(row_starts[i]), int(row_starts[i + 1])) for i in image_indices]
    return np.concatenate(row_lists)


def evaluate_feature_subset(X_raw: np.ndarray, y_all: np.ndarray, scaler, poly,
                             indices: np.ndarray, image_row_counts: list, n_folds: int = 5):
    # fold on images, not superpixel rows, to prevent spatial leakage
    n_images = len(image_row_counts)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2006)
    fold_ious = []

    for train_img_idx, val_img_idx in kf.split(np.arange(n_images)):
        train_rows = _rows_for_images(train_img_idx, image_row_counts)
        val_rows = _rows_for_images(val_img_idx, image_row_counts)

        X_tr = np.asarray(poly.transform(scaler.transform(X_raw[train_rows])))[:, indices]
        X_val = np.asarray(poly.transform(scaler.transform(X_raw[val_rows])))[:, indices]
        y_tr = y_all[train_rows]
        y_val = y_all[val_rows]

        xgb_fold = _fit_xgb_with_l1(X_tr, y_tr, reg_alpha=0.5, colsample_bytree=0.5)
        y_pred = xgb_fold.predict(X_val)

        # superpixel-level IoU is sufficient here; pixel-level is computed at test time
        TP = np.sum((y_pred == 1) & (y_val == 1))
        FP = np.sum((y_pred == 1) & (y_val == 0))
        FN = np.sum((y_pred == 0) & (y_val == 1))
        fold_ious.append(TP / (TP + FP + FN + 1e-10))

    return {'mean_iou': float(np.mean(fold_ious)), 'std_iou': float(np.std(fold_ious))}


def select_best_pruning(X_raw: np.ndarray, y_all: np.ndarray, checkpoints: list,
                        scaler, poly, image_row_counts: list) -> np.ndarray:
    if not checkpoints:
        raise RuntimeError("No pruning checkpoints to evaluate")

    print("\n  Evaluating pruning checkpoints via image-level CV:")
    best_iou: float = -1.0
    best_indices: np.ndarray = checkpoints[0]['indices'].copy()

    for cp in checkpoints:
        result = evaluate_feature_subset(X_raw, y_all, scaler, poly, cp['indices'], image_row_counts)
        print(f"    {cp['n_features']:3d} features → mean IoU {result['mean_iou']:.4f} ± {result['std_iou']:.4f}")
        if result['mean_iou'] > best_iou:
            best_iou = result['mean_iou']
            best_indices = cp['indices'].copy()

    print(f"  Selected subset: {len(best_indices)} features (IoU {best_iou:.4f})")
    return best_indices


def generate_trained_xgboost_poly(model_dir: str, image_folder_path: str):
    if image_folder_path is None:
        raise RuntimeError("Invalid image folder path")

    print("Building dataset")
    X_raw, y_all, image_row_counts = build_poly_dataset(image_folder_path)
    print(f"  {len(image_row_counts)} images, {len(y_all)} superpixels, {X_raw.shape[1]}-dim raw features")

    print("Fitting scaler and polynomial expansion...")
    scaler, poly, X_poly = fit_scaler_and_poly(X_raw)
    print(f"  Expanded to {X_poly.shape[1]} features")

    print("step 1, cheap filters")
    indices_s1 = apply_cheap_filters(X_poly)
    print(f"  {len(indices_s1)} features after variance + correlation filter")

    print("step 2, L1 XGB + gain trim")
    xgb_s2 = _fit_xgb_with_l1(X_poly[:, indices_s1], y_all, reg_alpha=1.0, colsample_bytree=0.4)
    keep_n_s2 = max(int(len(indices_s1) * 0.5), 30)
    indices_s2 = _gain_trim(xgb_s2, indices_s1, keep_n_s2)
    print(f"  {len(indices_s2)} features after gain trim")

    print("step 3, iterative pruning")
    checkpoints = iterative_prune(X_poly, y_all, indices_s2)

    print("step 4, image-level CV to select best cut depth...")
    best_indices = select_best_pruning(X_raw, y_all, checkpoints, scaler, poly, image_row_counts)

    print("Training final model on full training set")
    X_final = np.asarray(poly.transform(scaler.transform(X_raw)))[:, best_indices]
    scale_pos_weight = float(np.sum(y_all == 0)) / float(np.sum(y_all == 1))
    final_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=0.5,
        reg_lambda=1.0,
        colsample_bytree=0.5,
        colsample_bylevel=0.8,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=5,
        n_jobs=-1,
        random_state=2006,
        eval_metric="logloss",
    )
    final_xgb.fit(X_final, y_all)

    # bundle keeps scaler and poly as fitted objects — never refit at inference
    bundle = {
        'xgb': final_xgb,
        'scaler': scaler,
        'poly': poly,
        'selected_indices': best_indices,
    }
    save_model(bundle, model_dir, "xgboost_poly_model.joblib")
    print(f"Saved bundle ({len(best_indices)} features) to {model_dir}/xgboost_poly_model.joblib")
    return bundle


def load_xgboost_poly_bundle(model_dir: str):
    return load_model(model_dir, "xgboost_poly_model.joblib")


def single_image_predict_from_xgboost_poly(bundle: dict, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return None

    maps = extract_feature_maps(image)
    X = construct_superpixel_feature_matrix(
        maps["superpixel_labels"], maps["colour_maps"], maps["lbp_map"], maps["sobel_map"]
    )

    # preprocessing order must match training exactly: drop LBP bin → scale → poly → select
    X = np.delete(X, _LBP_REDUNDANT_BIN_INDEX, axis=1)
    X_transformed = np.asarray(bundle['poly'].transform(bundle['scaler'].transform(X)))[:, bundle['selected_indices']]

    y_pred = bundle['xgb'].predict(X_transformed)

    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    superpixel_labels = maps["superpixel_labels"]
    for index, label in enumerate(np.unique(superpixel_labels)):
        if y_pred[index] == 1:
            output_mask[superpixel_labels == label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_OPEN, kernel)
    output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_CLOSE, kernel)
    return output_mask


def predict_from_xgboost_poly(output_dir: str, bundle: dict, testing_folder: str):
    return predict_from_trained_model(output_dir, testing_folder, bundle, single_image_predict_from_xgboost_poly)
