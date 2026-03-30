# define our different ML models here
# leo's note, im gonna be trying random forests first
# other things could be interesting too

# we should probably compare deep learning methods to more traditiona ML pipelines in our write up
import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from .helpers import predict_from_trained_model, save_model, load_model
from .helpers import extract_feature_maps, construct_superpixel_feature_matrix, superpixel_label_from_ground_truth

def load_trained_randomforest(model_dir: str):
    return load_model(model_dir, "random_forest_model.joblib")

def generate_trained_randomforest(model_dir: str, image_folder_path: str):
    if image_folder_path is None:
        raise RuntimeError("Invalid image folder path")
    
    images_dir = Path(image_folder_path)

    X_all = []
    y_all = []

    for img in images_dir.glob("*.png"):

        if not img is None and not img.name.endswith("_mask.png"):
            result = training_randomforest_pipeline_on_image(str(img))
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

    # save model now
    save_model(random_forest, model_dir, "random_forest_model.joblib")

    return random_forest

def training_randomforest_pipeline_on_image(image_path: str):
    # read as bgr
    image = cv2.imread(image_path)
    binary_mask = cv2.imread(image_path.removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE)

    if image is None or binary_mask is None:
        return

    maps = extract_feature_maps(image)

    feature_matrix = construct_superpixel_feature_matrix(maps["superpixel_labels"], maps["colour_maps"], maps["lbp_map"], maps["sobel_map"])
    truth_map = superpixel_label_from_ground_truth(maps["superpixel_labels"], binary_mask)

    return feature_matrix, truth_map


def predict_from_trained_randomforest(output_dir: str,rf: RandomForestClassifier, testing_folder: str):
    return predict_from_trained_model(output_dir, testing_folder, rf, single_image_predict_from_trained_randomforest)
             

def single_image_predict_from_trained_randomforest(rf: RandomForestClassifier, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return

    maps = extract_feature_maps(image)
    feature_matrix = construct_superpixel_feature_matrix(maps["superpixel_labels"], maps["colour_maps"], maps["lbp_map"], maps["sobel_map"])

    y_pred = rf.predict(feature_matrix)

    output_mask = np.zeros(image.shape[:2], dtype = np.uint8)

    superpixel_labels = maps["superpixel_labels"]
    for index, label in enumerate(np.unique(superpixel_labels)):
        if y_pred[index] == 1:
            output_mask[superpixel_labels == label] = 255
    

    return output_mask


