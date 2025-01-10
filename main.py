from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_training_logs
import os
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def main():
    data_dir = "data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting training {timestamp}...")
    model, history = train_model(
        data_dir=data_dir,
        epochs=20,
        batch_size=32,
        learning_rate=0.001
    )

    # Save the training logs using our utility function
    save_training_logs(
        history, results_dir=results_dir, filename="training_logs.json")

    print("\n[INFO] Evaluating best model...")
    best_model_path = "saved_models/best_model.pth"

    if os.path.exists(best_model_path):
        precision, recall, f1, accuracy = evaluate_model(
            best_model_path, data_dir, plot_cm=True)
    else:
        print("[WARN] No best model found. Using final model...")
        precision, recall, f1, accuracy = evaluate_model(
            "saved_models/final_model.pth", data_dir, plot_cm=True)
    
    print(f"[METRICS] Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
