import cv2
import numpy as np
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold

from .helpers import predict_from_trained_model, save_model, load_model
from .helpers import extract_feature_maps, construct_superpixel_feature_matrix, superpixel_label_from_ground_truth


def load_trained_knn(model_dir: str):
    return load_model(model_dir, "knn_model.joblib")

def generate_trained_knn(model_dir: str, image_folder_path: str):
    if image_folder_path is None:
        raise RuntimeError("Invalid image folder path")

    images_dir = Path(image_folder_path)

    X_all = []
    y_all = []

    for img in images_dir.glob("*.png"):

        if not img is None and not img.name.endswith("_mask.png"):
            result = training_knn_pipeline_on_image(str(img))
            if result is None:
                continue
            f, t = result

            X_all.append(f)
            y_all.append(t)

    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)

    knn = KNeighborsClassifier(n_neighbors=9, weights="distance", p=1, n_jobs=1)
    knn.fit(X_train, y_train)

    # groups = np.concatenate([
    #     np.full(len(labels), img_idx) for img_idx, labels in enumerate(y_all)
    # ])

    # param_grid = {
    #     "n_neighbors": [9],
    #     "weights": ["distance"],
    #     "p": [1],
    # }
    #
    # search = GridSearchCV(
    #     KNeighborsClassifier(n_jobs=1),
    #     param_grid=param_grid,
    #     scoring="f1",
    #     cv=GroupKFold(n_splits=min(3, len(np.unique(groups)))),
    #     n_jobs=1,
    #     refit=True,
    #     verbose=3,
    # )
    #
    # search.fit(X_train, y_train, groups=groups)
    # knn = search.best_estimator_
    #
    # print("best params:", search.best_params_)
    # print("best f1:", search.best_score_)

    # save model now
    save_model(knn, model_dir, "knn_model.joblib")

    return knn

def training_knn_pipeline_on_image(image_path: str):
    # read as bgr
    image = cv2.imread(image_path)
    binary_mask = cv2.imread(image_path.removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE)

    if image is None or binary_mask is None:
        return

    maps = extract_feature_maps(image)

    feature_matrix = construct_superpixel_feature_matrix(maps["superpixel_labels"], maps["colour_maps"], maps["lbp_map"], maps["sobel_map"])
    truth_map = superpixel_label_from_ground_truth(maps["superpixel_labels"], binary_mask)

    return feature_matrix, truth_map


def predict_from_trained_knn(output_dir: str, knn: KNeighborsClassifier, testing_folder: str):
    return predict_from_trained_model(output_dir, testing_folder, knn, single_image_predict_from_trained_knn)


def single_image_predict_from_trained_knn(knn: KNeighborsClassifier, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return

    maps = extract_feature_maps(image)
    feature_matrix = construct_superpixel_feature_matrix(maps["superpixel_labels"], maps["colour_maps"],
                                                         maps["lbp_map"], maps["sobel_map"])

    y_pred = knn.predict(feature_matrix)

    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    superpixel_labels = maps["superpixel_labels"]
    for index, label in enumerate(np.unique(superpixel_labels)):
        if y_pred[index] == 1:
            output_mask[superpixel_labels == label] = 255


    return output_mask