from src.train import train_model
from src.evaluate import evaluate_model
import os

def main():
    data_dir = "data"
    
    print("Starting training...")
    model = train_model(
        data_dir=data_dir,
        epochs=20,
        batch_size=32,
        learning_rate=0.001
    )

    print("\nEvaluating best model...")
    best_model_path = "saved_models/best_model.pth"
    if os.path.exists(best_model_path):
        precision, recall, f1 = evaluate_model(best_model_path, data_dir)
    else:
        print("No best model found. Using final model...")
        precision, recall, f1 = evaluate_model("saved_models/final_model.pth", data_dir)

if __name__ == "__main__":
    main()
