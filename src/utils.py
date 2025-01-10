# utils.py

import os
import json
import datetime

def save_training_logs(history, results_dir="results", filename="training_logs.json"):
    """
    Saves the training 'history' dictionary (with losses/acc) as a JSON file in results_dir with a timestamp.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Incorporate prefix + timestamp
    logs_filename = f"{filename}_{timestamp}.json"
    logs_path = os.path.join(results_dir, logs_filename)

    with open(logs_path, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"[INFO] Training logs saved to: {logs_path}")
    return logs_path
