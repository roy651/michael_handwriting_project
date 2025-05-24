import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
import torchvision.models as models

import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')

#!/bin/bash
#!kaggle datasets download sujaymann/handwritten-english-characters-and-digits

#!unzip handwritten-english-characters-and-digits.zip

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class RandomChoice(torch.nn.Module):
    """Randomly applies one of the given transforms with given probability"""
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            transform = random.choice(self.transforms)
            return transform(img)
        return img
    
class ThicknessTransform:
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

class HandwritingDataPipeline:
    def __init__(self, data_root, image_size=64, batch_size=32,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
		         seed=42, device='mps', do_transform=False):
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if do_transform:
            # Define transforms with enhanced augmentations
            self.transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((64, 64)),
                
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
                
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), value=1.0)
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
                transforms.Resize((image_size, image_size)),  # Resize to standard size
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
            ])

        self.transform_eval = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load and split dataset
        self.setup_datasets(data_root, train_ratio, val_ratio, test_ratio)

    def setup_datasets(self, data_root, train_ratio, val_ratio, test_ratio):
        """Set up and split datasets with appropriate transforms"""
        # Same as before...
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"

        full_dataset = datasets.ImageFolder(root=data_root, transform=self.transform_eval)

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        lengths = [train_size, val_size, test_size]
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, lengths, generator=torch.Generator().manual_seed(self.seed)
        )

        train_dataset.dataset.transform = self.transform_train

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device != 'cpu' else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.sizes = {
            'train': train_size,
            'val': val_size,
            'test': test_size,
            'total': total_size
        }

    def get_loaders(self):
        """Return all three DataLoaders"""
        return self.train_loader, self.val_loader, self.test_loader

    def get_sizes(self):
        """Return the sizes of all splits"""
        return self.sizes

# Usage:
# pipeline = HandwritingDataPipeline(
#     data_root="./content/augmented_images/augmented_images1",
#     image_size=64,
#     batch_size=32,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     test_ratio=0.15
# )

# train_loader, val_loader, test_loader = pipeline.get_loaders()
# sizes = pipeline.get_sizes()

# print(f"Dataset splits: {sizes}")


def display_augmented_images(train_loader, num_images=5, num_augmentations=3):
    # Get class names from the dataset
    class_to_idx = train_loader.dataset.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Get random indices from the dataset
    dataset_size = len(train_loader.dataset)
    random_indices = torch.randperm(dataset_size)[:num_images]

    fig = plt.figure(figsize=(15, 3*num_images))

    for i, dataset_idx in enumerate(random_indices):
        # Get the original image and its true label
        original_path = train_loader.dataset.dataset.imgs[train_loader.dataset.indices[dataset_idx]][0]
        true_label = train_loader.dataset.dataset.imgs[train_loader.dataset.indices[dataset_idx]][1]
        class_name = idx_to_class[true_label]

        # Load and process original image
        original_pil = Image.open(original_path).convert('L')
        original_tensor = pipeline.transform_eval(original_pil)

        # Show true original image
        ax = plt.subplot(num_images, num_augmentations + 1, i*(num_augmentations + 1) + 1)
        img_display = original_tensor.cpu().numpy()[0]
        img_display = (img_display * 0.5) + 0.5  # Denormalize
        ax.imshow(img_display, cmap='gray')
        ax.set_title(f'Original (Class: {class_name})')
        ax.axis('off')

        # Show augmented versions
        for j in range(num_augmentations):
            aug_img = pipeline.transform_train(original_pil)

            ax = plt.subplot(num_images, num_augmentations + 1, i*(num_augmentations + 1) + j + 2)
            aug_display = aug_img.cpu().numpy()[0]
            aug_display = (aug_display * 0.5) + 0.5  # Denormalize
            ax.imshow(aug_display, cmap='gray')
            ax.set_title(f'Augmented {j+1} ({class_name})')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# Usage:
# display_augmented_images(train_loader, num_images=16, num_augmentations=8)


class LetterCNN64(nn.Module):
    def __init__(self, num_classes):
        super(LetterCNN64, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute correct flattened size dynamically
        self._to_linear = self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)  # Updated input size to (1, 64, 64)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.numel()  # Correct flattened size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImprovedLetterCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedLetterCNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Compute the flattened size dynamically
        self._to_linear = self._get_conv_output()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_conv_output(self):
        # Helper function to calculate the flattened size
        x = torch.zeros(1, 1, 64, 64)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG19HandwritingModel(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(VGG19HandwritingModel, self).__init__()
        
        # Load pretrained VGG19 and move to device
        vgg19 = models.vgg19(weights=('DEFAULT' if pretrained else None))
        vgg19 = vgg19.to(device)
        
        # Modify first layer to accept grayscale images
        self.features = vgg19.features
        self.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1).to(device)
        
        # Custom classifier for our task
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes)
        ).to(device)
        
        # Initialize weights for the new layers
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, save_dir='checkpoints'):
    train_losses, val_losses = [], []
    best_val_acc = 0.0
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_acc = 100 * correct / total

        # Step the scheduler with validation loss
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # Pass validation loss as metric
            else:
                scheduler.step()

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")

    # Save final model
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_acc': val_acc,
    }, os.path.join(save_dir, 'final_model.pth'))
    print(f"Saved final model at epoch {num_epochs}")

    # Plot training and validation loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return train_losses, val_losses, best_val_acc

# Function to load a saved model
def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    print(f"Loaded model from epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")
    return model, optimizer, epoch, val_acc


def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

def freeze_layers(model, num_layers_to_freeze):
    for i, param in enumerate(model.features.parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False

if __name__ == '__main__':
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = 'model_checkpoints'
    num_epochs=20
    pretrained=False

    # Create the data pipeline
    pipeline = HandwritingDataPipeline(
        data_root="./content/augmented_images/augmented_images1",
        image_size=64,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        device=device,
        do_transform=True
    )

    train_loader, val_loader, test_loader = pipeline.get_loaders()
    sizes = pipeline.get_sizes()
    print(f"Dataset splits: {sizes}")

    # display_augmented_images(train_loader, num_images=16, num_augmentations=8)

    # Get number of classes and initialize model
    num_classes = len(train_loader.dataset.dataset.classes)
    model = VGG19HandwritingModel(num_classes=num_classes, device=device, pretrained=pretrained)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)

    if pretrained:
        # Training configuration
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': 1e-5},  # Lower learning rate for VGG features
            {'params': model.classifier.parameters(), 'lr': 1e-4}  # Higher learning rate for new layers
        ])

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    else:
        num_epochs=50
        # Optimizer setup - single learning rate since we're training from scratch
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,  # Higher initial learning rate since we're training from scratch
            weight_decay=1e-4  # L2 regularization to prevent overfitting
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,  # Percentage of training to reach max_lr
            div_factor=10,  # Initial learning rate will be max_lr/div_factor
            final_div_factor=1e4  # Final learning rate will be max_lr/final_div_factor
        )        

    # Testing with different schedulers:
    # 1. Cosine Annealing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=num_epochs,
    #     eta_min=1e-6
    # )

    # 2. Step LR
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=5,
    #     gamma=0.5
    # )

    # # 3. One Cycle LR
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.01,
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_loader)
    # )

    # 4. Reduce LR on Plateau:
    # optimizer = optim.Adam(model.parameters(), lr=0.000005)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',      # Monitor loss (min) or metrics (max)
    #     factor=0.1,      # Reduction factor
    #     patience=3,      # Epochs to wait before reducing
    #     verbose=True,
    #     min_lr=1e-6
    # )

    # model, optimizer, epoch, val_acc = load_model(
    #     model,
    #     optimizer,
    #     os.path.join(save_dir, 'best_model.pth')
    # )

    # Train the model
    train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer,
                scheduler=scheduler, 
                num_epochs=num_epochs, 
                save_dir=save_dir
    )

    test_model(model, test_loader)

    # To load the best model later:
    model, optimizer, epoch, val_acc = load_model(
        model,
        optimizer,
        os.path.join(save_dir, 'best_model.pth')
    )
