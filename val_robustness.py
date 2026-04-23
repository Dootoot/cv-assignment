import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

os.environ["OMP_NUM_THREADS"] = "1"

from src.models.random_forest import load_trained_randomforest, single_image_predict_from_trained_randomforest
from src.models.xg_boosted_tree import load_trained_xgboost, single_image_predict_from_trained_xgboost
from src.models.unet import load_trained_unet, _single_image_predict as unet_predict
from src.models.transformer import load_trained_transformer, _single_image_predict as transformer_predict
from src.models.svm import load_trained_svm, single_image_predict_from_trained_svm
from src.models.fpn import load_trained_fpn, single_image_predict_from_trained_fpn as fpn_predict
from src.models.knn import load_trained_knn, single_image_predict_from_trained_knn
from src.models.mlp import load_trained_mlp, _single_image_predict as mlp_predict

from src.models.helpers import compute_metrics, save_prediction_comparison

load_dotenv()

model_dir = os.getenv("TRAINED_MODEL_PATH")
output_dir = os.getenv("MODEL_OUTPUT_PATH")

val_robustness_path = "data/EWS-Dataset/val_robustness"

conditions = {
    "gaussian_noise": "_gaussian_noise",
    "exposure": "_exposure",
    "blur": "_blur",
    "compression": "_compression",
}

all_suffixes = [
    "_gaussian_noise",
    "_exposure",
    "_blur",
    "_compression",
]

models = {
    "1": ("Random Forest", load_trained_randomforest, single_image_predict_from_trained_randomforest),
    "2": ("XGBoost", load_trained_xgboost, single_image_predict_from_trained_xgboost),
    "3": ("U-Net", load_trained_unet, unet_predict),
    "4": ("SegFormer", load_trained_transformer, transformer_predict),
    "5": ("SVM", load_trained_svm, single_image_predict_from_trained_svm),
    "6": ("FPN", load_trained_fpn, fpn_predict),
    "7": ("KNN", load_trained_knn, single_image_predict_from_trained_knn),
    "8": ("MLP", load_trained_mlp, mlp_predict),
}


def should_use_image(img: Path, condition_suffix):
    if img.name.endswith("_mask.png"):
        return False

    if condition_suffix is None:
        return not any(img.stem.endswith(suffix) for suffix in all_suffixes)

    return img.stem.endswith(condition_suffix)


def average_metrics(metrics_list):
    if not metrics_list:
        return {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "intersection_over_union": 0,
        }

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "f1_score": np.mean([m["f1_score"] for m in metrics_list]),
        "intersection_over_union": np.mean([m["intersection_over_union"] for m in metrics_list]),
    }


def test_condition(model_name, model, predict_func, condition_name, condition_suffix):
    images_dir = Path(val_robustness_path)
    condition_output_dir = (
        Path(output_dir)
        / "robustness_results"
        / model_name.replace(" ", "_")
        / condition_name
    )
    condition_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    count = 0

    for img in images_dir.glob("*.png"):
        if not should_use_image(img, condition_suffix):
            continue

        predicted_mask = predict_func(model, str(img))
        if predicted_mask is None:
            continue

        ground_truth = cv2.imread(
            str(img).removesuffix(".png") + "_mask.png",
            cv2.IMREAD_GRAYSCALE,
        )

        if ground_truth is None:
            print(f"Mask not found: {img.name}")
            continue

        metrics = compute_metrics(predicted_mask, ground_truth)
        metrics_list.append(metrics)

        save_prediction_comparison(
            predicted_mask,
            ground_truth,
            str(condition_output_dir),
            f"{img.stem}_predicted_against_truth.png",
        )

        count += 1

    avg = average_metrics(metrics_list)

    print("\n" + "=" * 60)
    print(f"{model_name} | {condition_name}")
    print(f"Images: {count}")
    print("=" * 60)
    print(avg)

    return avg


def test_one_model(model_name, load_fn, predict_func):
    model = load_fn(model_dir)

    if model is None:
        print(f"No saved {model_name} model found.")
        return

    for condition_name, condition_suffix in conditions.items():
        test_condition(
            model_name,
            model,
            predict_func,
            condition_name,
            condition_suffix,
        )


def main():
    while True:
        print("\nSelect model:")
        print("1. Random Forest")
        print("2. XGBoost")
        print("3. U-Net")
        print("4. SegFormer")
        print("5. SVM")
        print("6. FPN")
        print("7. KNN")
        print("8. MLP")
        print("9. Exit")

        choice = input("Enter choice: ")

        if choice == "9":
            break

        if choice not in models:
            print("Invalid choice.")
            continue

        model_name, load_fn, predict_func = models[choice]
        test_one_model(model_name, load_fn, predict_func)


if __name__ == "__main__":
    main()
