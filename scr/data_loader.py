import kagglehub
import os
import shutil

# Define the path to the dataset
data_folder = os.path.join(os.path.dirname(__file__), "..", "data")

# Ensure the "data" folder exists
os.makedirs(data_folder, exist_ok=True)

# Download the dataset
downloaded_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Move the downloaded dataset to the "data" folder
if os.path.exists(downloaded_path):
    shutil.move(downloaded_path, data_folder)
    print("Dataset successfully moved to:", data_folder)
else:
    print("Error: Dataset was not found at the downloaded path.")
