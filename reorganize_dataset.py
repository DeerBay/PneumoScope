import os
import random
import shutil
from pathlib import Path
from torchvision.transforms.functional import rotate, hflip
from PIL import Image

def augment_image(image_path, output_path):
    """
    Augment a single image by flipping and rotating, then save the augmented versions.
    """
    img = Image.open(image_path)
    augmented_images = [
        hflip(img),
        rotate(img, angle=10),
        rotate(img, angle=-10)
    ]
    for idx, aug_img in enumerate(augmented_images):
        aug_img.save(f"{output_path}_aug{idx}.jpeg")

def augment_random_images(input_dir, output_dir, num_to_augment=10):
    """
    Randomly select images from the input directory for augmentation.
    """
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    selected_images = random.sample(image_files, min(num_to_augment, len(image_files)))

    for image_file in selected_images:
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, os.path.splitext(image_file)[0])
        augment_image(image_path, output_path)


def reorganize_dataset(data_dir="data"):
    """
    Reorganize the dataset to have a better train/val/test split.
    Current split will be combined and then re-split to achieve:
    - Train: ~79% (4632 images)
    - Val: ~10% (600 images)
    - Test: ~11% (624 images) [unchanged]
    
    The function will:
    1. Keep test set as is
    2. Combine current train and val sets
    3. Split the combined set into new train and val sets
    4. Maintain class balance in the splits
    """
    # Create temporary directories for the reorganization
    temp_dir = Path(data_dir) / "temp"
    temp_normal = temp_dir / "NORMAL"
    temp_pneumonia = temp_dir / "PNEUMONIA"
    
    # Create temp directories
    temp_normal.mkdir(parents=True, exist_ok=True)
    temp_pneumonia.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Combine current train and val into temp
    print("Combining current train and val sets...")
    for split in ["train", "val"]:
        for class_name in ["NORMAL", "PNEUMONIA"]:
            src_dir = Path(data_dir) / split / class_name
            if src_dir.exists():
                for img in src_dir.glob("*.jpeg"):
                    shutil.copy2(str(img), str(temp_dir / class_name / img.name))
    
    # Count files
    normal_files = list(temp_normal.glob("*.jpeg"))
    pneumonia_files = list(temp_pneumonia.glob("*.jpeg"))
    
    print(f"\nTotal images in combined set:")
    print(f"NORMAL: {len(normal_files)}")
    print(f"PNEUMONIA: {len(pneumonia_files)}")
    
    # Calculate new val set sizes (approximately 10% of total)
    val_normal_size = 160  # ~10% of normal
    val_pneumonia_size = 440  # ~10% of pneumonia
    
    # Randomly select files for validation
    random.shuffle(normal_files)
    random.shuffle(pneumonia_files)
    
    val_normal = normal_files[:val_normal_size]
    val_pneumonia = pneumonia_files[:val_pneumonia_size]
    
    train_normal = normal_files[val_normal_size:]
    train_pneumonia = pneumonia_files[val_pneumonia_size:]
    
    # Remove existing train and val directories
    for split in ["train", "val"]:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            shutil.rmtree(str(split_dir))
    
    # Create new train and val directories
    for split in ["train", "val"]:
        for class_name in ["NORMAL", "PNEUMONIA"]:
            (Path(data_dir) / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Move files to new locations
    print("\nMoving files to new locations...")
    
    # Move validation files
    for img in val_normal:
        shutil.copy2(str(img), str(Path(data_dir) / "val" / "NORMAL" / img.name))
    for img in val_pneumonia:
        shutil.copy2(str(img), str(Path(data_dir) / "val" / "PNEUMONIA" / img.name))
    
    # Move training files
    for img in train_normal:
        shutil.copy2(str(img), str(Path(data_dir) / "train" / "NORMAL" / img.name))
    for img in train_pneumonia:
        shutil.copy2(str(img), str(Path(data_dir) / "train" / "PNEUMONIA" / img.name))
    
    # Clean up: remove temp directory
    shutil.rmtree(str(temp_dir))
    
    # Print final statistics
    print("\nNew dataset organization:")
    for split in ["train", "val", "test"]:
        split_path = Path(data_dir) / split
        if split_path.exists():
            normal_count = len(list((split_path / "NORMAL").glob("*.jpeg")))
            pneumonia_count = len(list((split_path / "PNEUMONIA").glob("*.jpeg")))
            total = normal_count + pneumonia_count
            print(f"\n{split.upper()}:")
            print(f"NORMAL: {normal_count}")
            print(f"PNEUMONIA: {pneumonia_count}")
            print(f"Total: {total}")
        else:
            print(f"\n{split.upper()}: Directory does not exist")

if __name__ == "__main__":
    reorganize_dataset()
    print("\nDataset reorganization complete!")
