import os
import shutil
import zipfile
import subprocess

# Define paths
data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
download_folder = os.path.join(os.path.dirname(__file__), "..", "downloads")
zip_file_path = os.path.join(download_folder, "chest-xray-pneumonia.zip")

# Ensure necessary folders exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(download_folder, exist_ok=True)

try:
    # Step 1: Download the dataset using Kaggle CLI
    print("Downloading dataset using Kaggle CLI...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia", "-p", download_folder],
        check=True
    )

    # Step 2: Extract the dataset
    print(f"Extracting dataset from {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_folder)

    # Step 3: Find the 'train', 'val', and 'test' folders and move them to the data folder
    extracted_root = os.path.join(download_folder, "chest_xray")
    folders_to_move = ['train', 'val', 'test']

    for folder in folders_to_move:
        source_path = os.path.join(extracted_root, folder)
        if os.path.exists(source_path):
            shutil.move(source_path, data_folder)
            print(f"Moved '{folder}' to '{data_folder}'.")

    print("Dataset successfully extracted and organized.")

finally:
    # Step 4: Cleanup - Remove the downloaded ZIP file and the downloads folder
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)
        print("Downloaded ZIP file removed.")

    if os.path.exists(download_folder):
        shutil.rmtree(download_folder)
        print("Downloads folder removed.")

print("Cleanup complete. All tasks finished.")