"""
Data utilities for handwritten character recognition.
Contains data pipelines, transformations, and augmentation techniques.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class RandomChoice(torch.nn.Module):
    """Apply one randomly chosen transform from a list of transforms."""
    
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            transform = random.choice(self.transforms)
            return transform(img)
        return img

    def __call__(self, img):
        return self.forward(img)


class ThicknessTransform:
    """Apply morphological operations to change stroke thickness using erosion/dilation."""
    
    def __init__(self, kernel_range=(1, 3), p=0.5):
        self.kernel_range = kernel_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # Convert PIL image to numpy array
            img_np = np.array(img)

            # Randomly choose operation (erosion or dilation)
            operation = np.random.choice(['erode', 'dilate'])

            # Random kernel size
            kernel_size = np.random.randint(self.kernel_range[0], self.kernel_range[1] + 1)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply morphological operation
            if operation == 'erode':
                processed = cv2.erode(img_np, kernel, iterations=1)
            else:
                processed = cv2.dilate(img_np, kernel, iterations=1)

            # Convert back to PIL
            return Image.fromarray(processed)
        return img


class ThicknessTransformAdvanced(torch.nn.Module):
    """Advanced thickness transform with more control over morphological operations."""
    
    def __init__(self, kernel_size=3, iterations=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, img):
        # Convert PIL Image to OpenCV format (numpy array)
        img_cv = np.array(img)
        
        # Ensure image is grayscale for morphological operations
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        elif len(img_cv.shape) == 3 and img_cv.shape[2] == 1:  # Already grayscale but 3-channel
             img_cv = img_cv[:, :, 0]
        
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Randomly choose between dilation (thicker) and erosion (thinner)
        if random.random() > 0.5:
            processed_img = cv2.dilate(img_cv, kernel, iterations=self.iterations)
        else:
            processed_img = cv2.erode(img_cv, kernel, iterations=self.iterations)
        
        # Convert back to PIL Image
        return Image.fromarray(processed_img, mode='L')  # 'L' for grayscale


class TransformedDataset(Dataset):
    """Helper class to apply transforms to subsets from random_split."""
    
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class HandwritingDataPipeline:
    """
    Comprehensive data pipeline for handwriting recognition.
    Handles loading, augmentation, and splitting of the image dataset.
    """
    
    def __init__(self, data_root, image_size=(64, 64), batch_size=32, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 seed=42, device='cpu', do_transform=False, 
                 normalization_type='imagenet'):
        """
        Initialize the data pipeline.
        
        Args:
            data_root (str): Root directory of the dataset
            image_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for DataLoaders
            train_ratio (float): Ratio for training split
            val_ratio (float): Ratio for validation split  
            test_ratio (float): Ratio for test split
            seed (int): Random seed for reproducibility
            device (str): Device for data loading
            do_transform (bool): Whether to apply data augmentation
            normalization_type (str): Type of normalization ('imagenet', 'simple', 'grayscale')
        """
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.do_transform = do_transform
        self.test_split = test_ratio
        self.val_split = val_ratio
        self.train_ratio = train_ratio
        self.seed = seed
        self.device = device
        self.normalization_type = normalization_type

        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Setup normalization based on type
        self._setup_normalization()
        
        # Setup transforms
        self._setup_transforms()
        
        # Load and split datasets
        self._load_and_split_datasets()

    def _setup_normalization(self):
        """Setup normalization parameters based on type."""
        if self.normalization_type == 'imagenet':
            # ImageNet normalization (good for transfer learning)
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        elif self.normalization_type == 'simple':
            # Simple normalization to [-1, 1]
            self.normalize = transforms.Normalize((0.5,), (0.5,))
        elif self.normalization_type == 'grayscale':
            # Grayscale normalization repeated for 3 channels
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def _setup_transforms(self):
        """Setup training and evaluation transforms."""
        if self.do_transform:
            # Enhanced training transforms with augmentation
            self.train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.Grayscale(num_output_channels=1),  # Ensure grayscale

                # Thickness variations through morphological operations
                ThicknessTransform(kernel_range=(1, 3), p=0.3),

                # Blur variations to simulate different pen pressures
                transforms.RandomApply([
                    transforms.GaussianBlur(3, sigma=(0.1, 0.2))
                ], p=0.3),

                # Conservative rotations
                transforms.RandomRotation(15, fill=255),

                # Gentle affine transformations
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=10,
                    fill=255
                ),

                # Perspective distortion
                transforms.RandomApply([
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
                ], p=0.3),

                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Repeat for 3 channels
                self.normalize,
                
                # Random erasing as final augmentation
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), value=1.0)
            ])
        else:
            # Simple training transform without augmentation
            self.train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                self.normalize
            ])

        # Evaluation transform (no augmentation)
        self.val_test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            self.normalize
        ])

    def _load_and_split_datasets(self):
        """Load the full dataset and split into train/val/test."""
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root directory '{self.data_root}' not found")
            
        try:
            full_dataset = datasets.ImageFolder(root=self.data_root)
            self.class_names = full_dataset.classes
            self.num_classes = len(self.class_names)
            
            if self.num_classes == 0:
                raise ValueError("No classes found in the dataset")

            total_size = len(full_dataset)
            if total_size == 0:
                raise ValueError("Dataset is empty")
                
            # Calculate split sizes
            train_size = int(self.train_ratio * total_size)
            val_size = int(self.val_ratio * total_size)
            test_size = total_size - train_size - val_size

            # Ensure all splits have at least one sample
            if train_size < 1 or val_size < 1 or test_size < 1:
                raise ValueError(f"Dataset too small ({total_size} samples) for the specified split ratios")

            print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}, Total={total_size}")

            # Perform the split
            train_temp_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size + val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_temp_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

            # Apply transforms to datasets
            self.train_dataset = TransformedDataset(train_dataset, transform=self.train_transform)
            self.val_dataset = TransformedDataset(val_dataset, transform=self.val_test_transform)
            self.test_dataset = TransformedDataset(test_dataset, transform=self.val_test_transform)
            
            self.sizes = {
                'train': len(self.train_dataset), 
                'val': len(self.val_dataset), 
                'test': len(self.test_dataset),
                'total': total_size
            }
            
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {str(e)}")

    def get_loaders(self, shuffle_train=True, shuffle_val=False, shuffle_test=False, num_workers=0):
        """
        Get DataLoaders for train, validation, and test sets.
        
        Args:
            shuffle_train (bool): Whether to shuffle training data
            shuffle_val (bool): Whether to shuffle validation data
            shuffle_test (bool): Whether to shuffle test data
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        pin_memory = True if self.device != 'cpu' else False
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle_train, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle_val, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle_test, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader

    def get_class_labels(self):
        """Get the class labels."""
        return self.class_names

    def get_sizes(self):
        """Get the dataset split sizes."""
        return self.sizes

    def get_sample_batch(self, split='train'):
        """
        Get a sample batch from the specified split.
        
        Args:
            split (str): Which split to sample from ('train', 'val', 'test')
            
        Returns:
            tuple: (images, labels) batch
        """
        if split == 'train':
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        elif split == 'val':
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        elif split == 'test':
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            raise ValueError(f"Unknown split: {split}")
            
        return next(iter(loader))


def get_class_labels_from_directory(data_root):
    """
    Get class labels from directory structure.
    
    Args:
        data_root (str): Root directory path
        
    Returns:
        list: List of class names
    """
    if not os.path.exists(data_root):
        print(f"Error: Data root directory '{data_root}' not found")
        return []
        
    try:
        # Get subdirectory names as class labels
        class_names = sorted([d for d in os.listdir(data_root) 
                            if os.path.isdir(os.path.join(data_root, d)) 
                            and not d.startswith('.')])
        return class_names
    except Exception as e:
        print(f"Error loading class labels from '{data_root}': {e}")
        return []


def prepare_single_image(image_path, image_size=(64, 64), normalization_type='imagenet'):
    """
    Prepare a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        image_size (tuple): Target image size
        normalization_type (str): Type of normalization
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found")
    
    # Setup normalization
    if normalization_type == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif normalization_type == 'simple':
        normalize = transforms.Normalize((0.5,), (0.5,))
    elif normalization_type == 'grayscale':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")
    
    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        normalize
    ])
    
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise RuntimeError(f"Error processing image '{image_path}': {str(e)}")


def create_custom_transforms(augmentation_level='medium'):
    """
    Create custom transforms based on augmentation level.
    
    Args:
        augmentation_level (str): Level of augmentation ('light', 'medium', 'heavy')
        
    Returns:
        transforms.Compose: Transform pipeline
    """
    base_transforms = [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1)
    ]
    
    augment_transforms = []
    
    if augmentation_level == 'light':
        augment_transforms = [
            transforms.RandomRotation(5, fill=255),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.2))], p=0.2)
        ]
    elif augmentation_level == 'medium':
        augment_transforms = [
            ThicknessTransform(kernel_range=(1, 3), p=0.3),
            transforms.RandomRotation(15, fill=255),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.2))], p=0.3)
        ]
    elif augmentation_level == 'heavy':
        augment_transforms = [
            ThicknessTransform(kernel_range=(1, 5), p=0.4),
            transforms.RandomRotation(20, fill=255),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15, fill=255),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.4, p=0.5)], p=0.4),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.3))], p=0.4),
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ]
    
    final_transforms = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if augmentation_level in ['medium', 'heavy']:
        final_transforms.insert(-1, transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), value=1.0))
    
    return transforms.Compose(base_transforms + augment_transforms + final_transforms)
