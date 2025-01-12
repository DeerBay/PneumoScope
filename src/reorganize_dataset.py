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
    """Count images by type (normal, virus, bacteria) in each split (OLD style)."""
    from collections import defaultdict
    counts = defaultdict(int)
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
                counts[(dataset_type, "normal")] += len(os.listdir(label_path))
            elif label == "PNEUMONIA":
                for image_name in os.listdir(label_path):
                    if "bacteria" in image_name.lower():
                        counts[(dataset_type, "bacteria")] += 1
                    elif "virus" in image_name.lower():
                        counts[(dataset_type, "virus")] += 1
    return counts

def print_dataset_statistics(data_dir, title="Dataset Statistics"):
    """Print detailed statistics about the dataset (binary style)."""
    counts = count_images_by_type(data_dir)
    # Convert to DataFrame
    from collections import OrderedDict
    out_dict = {}
    for k, v in counts.items():
        out_dict[k] = v
    df = pd.DataFrame.from_dict(out_dict, orient='index', columns=['Image Count'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Set", "Type"])

    # Calculate total
    total_normal = sum(counts[(split, "normal")] for split in ['train','test','val'] if (split, "normal") in counts)
    total_bacteria = sum(counts[(split, "bacteria")] for split in ['train','test','val'] if (split, "bacteria") in counts)
    total_virus = sum(counts[(split, "virus")] for split in ['train','test','val'] if (split, "virus") in counts)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Total images by type:")
    print(f"NORMAL: {total_normal}")
    print(f"BACTERIA: {total_bacteria}")
    print(f"VIRUS: {total_virus}")
    print(f"\nDetailed distribution:")
    print(df)

def reorganize_dataset(data_dir="data", val_size=0.15, test_size=0.15):
    """
    Reorganize the dataset in two ways:
      1) In the original 'data/' folder, create train/val/test with NORMAL / PNEUMONIA.
      2) In a new 'data_multi/' folder, create train/val/test with NORMAL / BACTERIA / VIRUS.
    """
    print("[INFO] Starting dataset reorganization for binary + multiclass...")

    # ========== PART A) Binary setup in `data/` ==========

    print("\n[INFO] First step: reorganize 'data/' into NORMAL/PNEUMONIA.")
    print("Original dataset statistics:")
    print_dataset_statistics(data_dir, "Original Dataset Statistics")

    # Create temp
    from pathlib import Path
    temp_dir = Path(data_dir) / "temp"
    temp_normal = temp_dir / "NORMAL"
    temp_bacteria = temp_dir / "PNEUMONIA_BACTERIA"
    temp_virus = temp_dir / "PNEUMONIA_VIRUS"

    for d_ in [temp_normal, temp_bacteria, temp_virus]:
        d_.mkdir(parents=True, exist_ok=True)

    # Combine
    for split in ["train","val","test"]:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            normal_dir = split_dir / "NORMAL"
            if normal_dir.exists():
                for img in normal_dir.glob("*.jpeg"):
                    shutil.copy2(str(img), str(temp_normal / img.name))

            pneumonia_dir = split_dir / "PNEUMONIA"
            if pneumonia_dir.exists():
                for img in pneumonia_dir.glob("*.jpeg"):
                    ptype = get_pneumonia_type(img.name)
                    if ptype == "bacteria":
                        shutil.copy2(str(img), str(temp_bacteria / img.name))
                    elif ptype == "virus":
                        shutil.copy2(str(img), str(temp_virus / img.name))

    # Shuffle
    import random
    normal_files = list(temp_normal.glob("*.jpeg"))
    bacteria_files = list(temp_bacteria.glob("*.jpeg"))
    virus_files = list(temp_virus.glob("*.jpeg"))

    random.shuffle(normal_files)
    random.shuffle(bacteria_files)
    random.shuffle(virus_files)

    def split_files(files, val_size, test_size):
        n = len(files)
        val_n = int(n*val_size)
        test_n = int(n*test_size)
        train_n = n - val_n - test_n
        return {
            'train': files[:train_n],
            'val': files[train_n:train_n+val_n],
            'test': files[train_n+val_n:]
        }

    normal_split = split_files(normal_files, val_size, test_size)
    bacteria_split = split_files(bacteria_files, val_size, test_size)
    virus_split = split_files(virus_files, val_size, test_size)

    # Remove existing subfolders in data
    for split in ["train","val","test"]:
        sub_ = Path(data_dir) / split
        if sub_.exists():
            shutil.rmtree(str(sub_))

    # Create new
    for split in ["train","val","test"]:
        (Path(data_dir)/split/"NORMAL").mkdir(parents=True, exist_ok=True)
        (Path(data_dir)/split/"PNEUMONIA").mkdir(parents=True, exist_ok=True)

    # Move
    print("\n[INFO] Rebuilding 'data/' for binary classification (NORMAL/PNEUMONIA)...")
    for split in ["train","val","test"]:
        # normal
        for img in normal_split[split]:
            shutil.copy2(str(img), str(Path(data_dir)/split/"NORMAL"/img.name))
        # pneumonia => bacteria + virus
        for img in bacteria_split[split] + virus_split[split]:
            shutil.copy2(str(img), str(Path(data_dir)/split/"PNEUMONIA"/img.name))

    # Clean temp
    shutil.rmtree(str(temp_dir))

    print("\n[INFO] New dataset statistics (binary):")
    print_dataset_statistics(data_dir, "New Dataset Statistics")

    # ========== PART B) Multiclass setup in `data_multi/` ==========

    print("\n[INFO] Now create 'data_multi/' folder with NORMAL/BACTERIA/VIRUS.")
    # We'll do a similar approach. Create data_multi with train/val/test subfolders.

    data_multi_dir = Path(data_dir).parent / "data_multi"
    if data_multi_dir.exists():
        shutil.rmtree(str(data_multi_dir))
    data_multi_dir.mkdir(parents=True, exist_ok=True)

    # temp for multi
    temp_multi = data_multi_dir / "temp_multi"
    temp_multi.mkdir(parents=True, exist_ok=True)
    temp_m_normal = temp_multi / "NORMAL"
    temp_m_bact   = temp_multi / "BACTERIA"
    temp_m_virus  = temp_multi / "VIRUS"
    temp_m_normal.mkdir(parents=True, exist_ok=True)
    temp_m_bact.mkdir(parents=True, exist_ok=True)
    temp_m_virus.mkdir(parents=True, exist_ok=True)

    for split in ["train","val","test"]:
        p_normal = Path(data_dir)/split/"NORMAL"
        p_pneum  = Path(data_dir)/split/"PNEUMONIA"
        if p_normal.exists():
            for img in p_normal.glob("*.jpeg"):
                shutil.copy2(str(img), str(temp_m_normal / img.name))
        if p_pneum.exists():
            for img in p_pneum.glob("*.jpeg"):
                ptype = get_pneumonia_type(img.name)
                if ptype == "bacteria":
                    shutil.copy2(str(img), str(temp_m_bact / img.name))
                elif ptype == "virus":
                    shutil.copy2(str(img), str(temp_m_virus / img.name))

    # Shuffle for multi
    m_normal_files = list(temp_m_normal.glob("*.jpeg"))
    m_bact_files   = list(temp_m_bact.glob("*.jpeg"))
    m_virus_files  = list(temp_m_virus.glob("*.jpeg"))

    random.shuffle(m_normal_files)
    random.shuffle(m_bact_files)
    random.shuffle(m_virus_files)

    normal_splt = split_files(m_normal_files, val_size, test_size)
    bact_splt   = split_files(m_bact_files, val_size, test_size)
    virus_splt  = split_files(m_virus_files, val_size, test_size)

    # create subfolders under data_multi
    for split in ["train","val","test"]:
        (data_multi_dir / split / "NORMAL").mkdir(parents=True, exist_ok=True)
        (data_multi_dir / split / "BACTERIA").mkdir(parents=True, exist_ok=True)
        (data_multi_dir / split / "VIRUS").mkdir(parents=True, exist_ok=True)

    # Move them
    print("[INFO] Building 'data_multi' with NORMAL/BACTERIA/VIRUS...")
    for split in ["train","val","test"]:
        for img in normal_splt[split]:
            shutil.copy2(str(img), str(data_multi_dir/split/"NORMAL"/img.name))
        for img in bact_splt[split]:
            shutil.copy2(str(img), str(data_multi_dir/split/"BACTERIA"/img.name))
        for img in virus_splt[split]:
            shutil.copy2(str(img), str(data_multi_dir/split/"VIRUS"/img.name))

    # remove temp_multi
    shutil.rmtree(str(temp_multi))

    print("[INFO] All done. 'data/' => binary, 'data_multi/' => multiclass.")


if __name__ == "__main__":
    reorganize_dataset()
    print("\nDataset reorganization complete!")
