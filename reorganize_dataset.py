import os
import random
import shutil
from pathlib import Path
from collections import Counter
import pandas as pd
import re

def get_pneumonia_type(filename):
    """Extract whether a pneumonia case is bacterial or viral from filename."""
    if "bacteria" in filename.lower():
        return "bacteria"
    elif "virus" in filename.lower():
        return "virus"
    return None

def count_images_by_type(base_path):
    """Count images by type (normal, virus, bacteria) in each split."""
    counts = Counter()
    for dataset_type in ['train', 'test', 'val']:
        dataset_path = os.path.join(base_path, dataset_type)
        if not os.path.exists(dataset_path):
            continue
            
        for label in ['NORMAL', 'PNEUMONIA']:
            label_path = os.path.join(dataset_path, label)
            if not os.path.exists(label_path):
                continue
                
            if label == "NORMAL":
                # Count all images in the NORMAL folder as "normal"
                counts[(dataset_type, "normal")] = len(os.listdir(label_path))
            elif label == "PNEUMONIA":
                for image_name in os.listdir(label_path):
                    if "virus" in image_name.lower():
                        counts[(dataset_type, "virus")] += 1
                    elif "bacteria" in image_name.lower():
                        counts[(dataset_type, "bacteria")] += 1
    return counts

def print_dataset_statistics(data_dir, title="Dataset Statistics"):
    """Print detailed statistics about the dataset."""
    counts = count_images_by_type(data_dir)
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['Image Count'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Set", "Type"])
    
    # Calculate total counts per type
    total_normal = sum(counts[(split, "normal")] for split in ['train', 'test', 'val'] if (split, "normal") in counts)
    total_bacteria = sum(counts[(split, "bacteria")] for split in ['train', 'test', 'val'] if (split, "bacteria") in counts)
    total_virus = sum(counts[(split, "virus")] for split in ['train', 'test', 'val'] if (split, "virus") in counts)
    
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Total images by type:")
    print(f"NORMAL: {total_normal}")
    print(f"BACTERIA: {total_bacteria}")
    print(f"VIRUS: {total_virus}")
    print(f"\nDetailed distribution:")
    print(df)
    
    # Calculate percentages within each set
    for split in ['train', 'test', 'val']:
        if any((split, t) in counts for t in ["normal", "bacteria", "virus"]):
            split_total = sum(counts[(split, t)] for t in ["normal", "bacteria", "virus"] if (split, t) in counts)
            print(f"\nPercentage distribution in {split}:")
            for type_ in ["normal", "bacteria", "virus"]:
                if (split, type_) in counts:
                    percentage = (counts[(split, type_)] / split_total) * 100
                    print(f"{type_}: {percentage:.1f}%")

def reorganize_dataset(data_dir="data", val_size=0.15, test_size=0.15):
    """
    Reorganize the dataset to have a balanced train/val/test split.
    
    Args:
        data_dir (str): Path to data directory
        val_size (float): Proportion of data for validation (0-1)
        test_size (float): Proportion of data for testing (0-1)
    """
    print("Original dataset statistics:")
    print_dataset_statistics(data_dir, "Original Dataset Statistics")
    
    # Create temporary directories for the reorganization
    temp_dir = Path(data_dir) / "temp"
    temp_normal = temp_dir / "NORMAL"
    temp_bacteria = temp_dir / "PNEUMONIA_BACTERIA"
    temp_virus = temp_dir / "PNEUMONIA_VIRUS"
    
    # Create temp directories
    for dir_path in [temp_normal, temp_bacteria, temp_virus]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Combine all sets into temp
    print("\nCombining all sets...")
    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        
        # Handle normal cases
        normal_dir = split_dir / "NORMAL"
        if normal_dir.exists():
            for img in normal_dir.glob("*.jpeg"):
                shutil.copy2(str(img), str(temp_normal / img.name))
        
        # Handle pneumonia cases (separate bacteria and virus)
        pneumonia_dir = split_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img in pneumonia_dir.glob("*.jpeg"):
                ptype = get_pneumonia_type(img.name)
                if ptype == "bacteria":
                    shutil.copy2(str(img), str(temp_bacteria / img.name))
                elif ptype == "virus":
                    shutil.copy2(str(img), str(temp_virus / img.name))
    
    # Get file lists
    normal_files = list(temp_normal.glob("*.jpeg"))
    bacteria_files = list(temp_bacteria.glob("*.jpeg"))
    virus_files = list(temp_virus.glob("*.jpeg"))
    
    # Shuffle files
    random.shuffle(normal_files)
    random.shuffle(bacteria_files)
    random.shuffle(virus_files)
    
    # Calculate split sizes
    def split_files(files, val_size, test_size):
        n = len(files)
        val_n = int(n * val_size)
        test_n = int(n * test_size)
        train_n = n - val_n - test_n
        return {
            'train': files[:train_n],
            'val': files[train_n:train_n + val_n],
            'test': files[train_n + val_n:]
        }
    
    # Split each class
    normal_split = split_files(normal_files, val_size, test_size)
    bacteria_split = split_files(bacteria_files, val_size, test_size)
    virus_split = split_files(virus_files, val_size, test_size)
    
    # Remove existing directories
    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            shutil.rmtree(str(split_dir))
    
    # Create new directories
    for split in ["train", "val", "test"]:
        (Path(data_dir) / split / "NORMAL").mkdir(parents=True, exist_ok=True)
        (Path(data_dir) / split / "PNEUMONIA").mkdir(parents=True, exist_ok=True)
    
    # Move files to new locations
    print("\nMoving files to new locations...")
    
    for split in ["train", "val", "test"]:
        # Move normal files
        for img in normal_split[split]:
            shutil.copy2(str(img), str(Path(data_dir) / split / "NORMAL" / img.name))
        
        # Move bacteria and virus files
        for img in bacteria_split[split] + virus_split[split]:
            shutil.copy2(str(img), str(Path(data_dir) / split / "PNEUMONIA" / img.name))
    
    # Clean up: remove temp directory
    shutil.rmtree(str(temp_dir))
    
    print("\nNew dataset statistics:")
    print_dataset_statistics(data_dir, "New Dataset Statistics")

if __name__ == "__main__":
    reorganize_dataset()
    print("\nDataset reorganization complete!")
