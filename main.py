import os
import time
from dotenv import load_dotenv
from src.models.random_forest import load_trained_randomforest, generate_trained_randomforest, predict_from_trained_randomforest
from src.models.xg_boosted_tree import load_trained_xgboost, generate_trained_xgboost, predict_from_trained_xgboost

load_dotenv()

model_dir = os.getenv("TRAINED_MODEL_PATH")
train_path = os.getenv("TRAINING_FOLDER_PATH")
val_path = os.getenv("VALIDATION_FOLDER_PATH")
output_dir = os.getenv("MODEL_OUTPUT_PATH")

if model_dir is None or train_path is None or val_path is None or output_dir is None:
    raise RuntimeError("One or more required environment variables are not set. Please check your .env file.")

feature_names = [
    "lab_l", "lab_a", "lab_b", "h", "s", "v", "r", "g", "b", "exG",
    "std_lab_a", "std_s", "std_g", "std_exG",
    "lbp_0", "lbp_1", "lbp_2", "lbp_3", "lbp_4",
    "lbp_5", "lbp_6", "lbp_7", "lbp_8", "lbp_9",
    "sobel_mean", "sobel_std",
    "area", "eccentricity", "compactness"
]

def print_feature_importances(model):
    print("\n--- Feature Importances ---")
    for name, score in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")

def train_or_test_prompt(model_name, load_fn, train_fn, predict_fn, model_filename):
    print(f"\n{'='*60}")
    print(f"{model_name} selected")
    print(f"{'='*60}")
    print("1. Train new model (overwrites existing)")
    print("2. Test existing model on validation set")
    print("3. Back")

    choice = input("Enter your choice (1/2/3): ")

    match choice:
        case "1":
            # delete existing model file if present
            model_path = os.path.join(str(model_dir), model_filename)
            if os.path.exists(model_path):
                os.remove(model_path)
                print("Removed old saved model.")
            print(f"Training {model_name}...")

            start = time.time()
            model = train_fn(model_dir, train_path)
            print(f"Training time: {time.time() - start:.2f}s")
            print_feature_importances(model)

            # also run validation after training
            start = time.time()
            metrics = predict_fn(output_dir, model, val_path)
            print(f"Validation time: {time.time() - start:.2f}s")
            print(metrics)

        case "2":
            model = load_fn(model_dir)
            if model is None:
                print(f"No saved {model_name} model found. Please train first.")
            else:
                print(f"Loaded saved {model_name} model.")
                print_feature_importances(model)

                start = time.time()
                metrics = predict_fn(output_dir, model, val_path)
                print(f"Validation time: {time.time() - start:.2f}s")
                print(metrics)

        case "3":
            return
        case _:
            print("Invalid choice.")

def main_menu():
    while True:
        print(f"\n{'='*60}")
        print("Select a model:")
        print(f"{'='*60}")
        print("1. Random Forest")
        print("2. XGBoost")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        match choice:
            case "1":
                train_or_test_prompt(
                    "Random Forest",
                    load_trained_randomforest,
                    generate_trained_randomforest,
                    predict_from_trained_randomforest,
                    "random_forest_model.joblib"
                )
            case "2":
                train_or_test_prompt(
                    "XGBoost",
                    load_trained_xgboost,
                    generate_trained_xgboost,
                    predict_from_trained_xgboost,
                    "xgboost_model.joblib"
                )
            case "3":
                print("Exiting...")
                break
            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
