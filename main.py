from src.train import train_model
from src.evaluate import evaluate_model
import os
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def main():
    """Main training and evaluation pipeline"""
    # Setup directories relative to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    save_dir = os.path.join(base_dir, "saved_models")
    results_dir = os.path.join(base_dir, "results")
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting training {timestamp}...")
    model, best_model_path, logs_path = train_model(
        data_dir=data_dir,
        save_dir=save_dir,
        results_dir=results_dir
    )

    print("\n[INFO] Evaluating best model...")
    if os.path.exists(best_model_path):
        precision, recall, f1, accuracy = evaluate_model(
            model_path=best_model_path,
            data_dir=data_dir,
            results_dir=results_dir,
            plot_cm=True
        )
        print(f"\n[FINAL METRICS]")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
    else:
        print(f"[ERROR] Best model not found at: {best_model_path}")

if __name__ == "__main__":
    main()
