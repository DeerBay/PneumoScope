from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def get_data_loaders(data_dir, batch_size=32,
                     augment_train=True,
                     random_crop=False,
                     color_jitter=False,
                     num_workers=2, balance_train=True, desired_total_samples=None):
    """
    Get DataLoaders for train/val/test.
    :param augment_train: (bool) whether to apply random flips, rotation etc. to train
    :param random_crop: (bool) whether to do random crop on train
    :param color_jitter: (bool) whether to apply color jitter
    :param num_workers: (int) number of worker processes for data loading
    """
    
    # Base transform list for "train"
    train_transform_list = []
    
    # 1) Resize (e.g., 224x224 for ResNet)
    if random_crop:
        # First Resize then Crop
        train_transform_list.append(transforms.Resize((256, 256)))
        train_transform_list.append(transforms.RandomCrop((224,224)))
    else:
        # No crop, just resize
        train_transform_list.append(transforms.Resize((224, 224)))
    
    # 2) Augmentations
    if augment_train:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        train_transform_list.append(transforms.RandomRotation(degrees=15))
        if color_jitter:
            train_transform_list.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
    
    # 3) Convert to Tensor & Normalize
    #   ResNet18 expects normalization according to ImageNet statistics
    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))

    transform_train = transforms.Compose(train_transform_list)

    # Val / test transform
    transform_valtest = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load datasets
    train_data = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform_train)
    val_data   = datasets.ImageFolder(root=f'{data_dir}/val',   transform=transform_valtest)
    test_data  = datasets.ImageFolder(root=f'{data_dir}/test',  transform=transform_valtest)


    # Calculate the total samples for the sampler
    if desired_total_samples is None:
        # Default: same number of samples as in the dataset
        desired_total_samples = len(sample_weights)

        # Oversampling with WeightedRandomSampler
    if balance_train:
        targets = [label for _, label in train_data.samples]
        class_sample_counts = np.bincount(targets)
        class_weights = 1.0 / class_sample_counts
        sample_weights = [class_weights[label] for label in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
