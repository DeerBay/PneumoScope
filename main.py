import os
import sys

# Kaggle downloader
from src.kaggle_downloader import download_dataset_if_needed  # see below for the updated function

# For dataset reorganization:
from src.reorganize_dataset import reorganize_dataset

# For binary training & evaluation
from src.train import train_model
from src.evaluate import evaluate_model

# For multiclass training & evaluation
from src.train_multiclass import train_model_multiclass
from src.evaluate_multiclass import evaluate_model_multiclass


def main():
    """
    Main pipeline:
      0) If the raw data doesn't exist, run Kaggle downloader
      1) If reorganized data doesn't exist, reorganize
      2) Train/eval binary
      3) Train/eval multiclass
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 0) Check if the main data "train/val/test" do NOT exist, 
    #    then run Kaggle downloader automatically
    data_dir = os.path.join(base_dir, "data")
    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")
    test_path  = os.path.join(data_dir, "test")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("[INFO] 'train/', 'val/', or 'test/' folder not found - running Kaggle downloader...")
        download_dataset_if_needed()  # We'll define an updated function in kaggle_downloader.py
    else:
        print("[INFO] Found existing 'train/', 'val/', 'test/' inside 'data/'. Skipping Kaggle download.")

    # 1) Check if data_multi/ exists. If not, reorganize
    data_multi_dir = os.path.join(base_dir, "data_multi")
    if not os.path.exists(data_multi_dir):
        print("[INFO] data_multi/ folder not found - reorganizing for both binary & multiclass...")
        reorganize_dataset(data_dir=data_dir)  # This also creates data_multi/
    else:
        print("[INFO] data_multi/ already exists. Skipping reorganize_dataset.")

    # 2) Train binary
    data_dir_binary = data_dir  # 'data' folder for binary
    save_dir_binary = os.path.join(base_dir, "saved_models")  # inside is normal
    results_dir_binary = os.path.join(base_dir, "results")

    print("\n[INFO] Starting BINARY training...")
    _, best_bin_path, _ = train_model(
        data_dir=data_dir_binary,
        save_dir=save_dir_binary,
        results_dir=results_dir_binary
    )

    # Evaluate binary
    print("\n[INFO] Evaluating BINARY model...")
    if os.path.exists(best_bin_path):
        precision, recall, f1, accuracy, auc_score, avg_precision = evaluate_model(
            model_path=best_bin_path,
            data_dir=data_dir_binary,
            results_dir=results_dir_binary,
            plot_cm=False
        )
        print(f"[BINARY] Precision={precision:.4f}, Recall={recall:.4f}, "
              f"F1={f1:.4f}, Acc={accuracy:.4f}, AUC={auc_score:.4f}")
    else:
        print("[ERROR] Best binary model not found at:", best_bin_path)
        print("[ERROR] Cannot proceed without binary model evaluation.")
        sys.exit(1)  # Exit with error code

    # 3) Train multiclass
    data_dir_multi = data_multi_dir  # 'data_multi' folder for multiclass
    save_dir_multi = os.path.join(base_dir, "saved_models", "multiclass_models")
    results_dir_multi = os.path.join(base_dir, "results", "multiclass")

    print("\n[INFO] Starting MULTICLASS training...")
    _, best_multi_path, _ = train_model_multiclass(
        data_dir=data_dir_multi,
        save_dir=save_dir_multi,
        results_dir=results_dir_multi
    )

    # Evaluate multiclass
    print("\n[INFO] Evaluating MULTICLASS model...")
    if os.path.exists(best_multi_path):
        precision_m, recall_m, f1_m, acc_m = evaluate_model_multiclass(
            model_path=best_multi_path,
            data_dir=data_dir_multi,
            results_dir=results_dir_multi
        )
        print(f"[MULTICLASS] Precision(macro)={precision_m:.4f}, "
              f"Recall(macro)={recall_m:.4f}, F1(macro)={f1_m:.4f}, Acc={acc_m:.4f}")
    else:
        print("[ERROR] Best multiclass model not found at:", best_multi_path)
        print("[ERROR] Cannot proceed without multiclass model evaluation.")
        sys.exit(1)  # Exit with error code

    print("\n[INFO] Done. Both binary and multiclass training/evaluation completed.")


if __name__ == "__main__":
    main()
