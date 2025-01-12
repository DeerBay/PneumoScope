from src.hyperparameter_tuning import grid_search
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_training_logs
import os
import datetime



def main(tune_hyperparameters=False):
    data_dir = "data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if tune_hyperparameters:
        print(f"Starting hyperparameter search {timestamp}...")
        best_params, results = grid_search(data_dir)

        print("\nBest parameters found:")
        print(best_params)

        # Save hyperparameter search results
        with open(f"{results_dir}/hyperparameter_search_{timestamp}.txt", "w") as f:
            f.write("Hyperparameter Search Results\n")
            f.write("===========================\n\n")
            f.write(f"Best parameters: {best_params}\n\n")
            f.write("All results:\n")
            for result in sorted(results, key=lambda x: x['val_acc'], reverse=True):
                f.write(f"\nParams: {result['params']}")
                f.write(f"\nVal Accuracy: {result['val_acc']:.4f}\n")

        # Train final model with best parameters
        print(f"\nTraining final model with best parameters...")
        model, history = train_model(
            data_dir=data_dir,
            **best_params
        )
    else:
        print(f"Starting training {timestamp}...")
        model, history = train_model(
            data_dir=data_dir,
            epochs=2, # 20
            batch_size=32,
            learning_rate=0.001,
            patience=5,        # Early stopping patience
            lr_patience=3,     # Reduce LR after 3 epochs without improvement
            lr_factor=0.1      # Reduce LR by factor of 10
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
