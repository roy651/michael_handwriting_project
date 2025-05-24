"""
Model architectures for handwritten character recognition.
Contains custom CNN models and transfer learning implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LetterCNN64(nn.Module):
    """Basic CNN model designed for 64x64 pixel input images."""
    
    def __init__(self, num_classes):
        super(LetterCNN64, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input 3 channels (RGB-like after transform)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input size: 64x64 -> After conv1 (padding=1): 64x64 -> After pool1: 32x32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input size: 32x32 -> After conv2 (padding=1): 32x32 -> After pool2: 16x16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input size: 16x16 -> After conv3 (padding=1): 16x16 -> After pool3: 8x8
        
        # Fully connected layers
        # Flattened size: 128 channels * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten the tensor
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


class ImprovedLetterCNN(nn.Module):
    """Enhanced CNN with Batch Normalization and Dropout for better performance."""
    
    def __init__(self, num_classes):
        super(ImprovedLetterCNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64x64 -> 32x32

        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32x32 -> 16x16

        # Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16x16 -> 8x8

        # Layer 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 8x8 -> 4x4

        # Fully connected layers
        # Flattened size: 256 channels * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Dropout for regularization

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # FC layers
        x = self.dropout1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x


class VGG19HandwritingModel(nn.Module):
    """VGG19-based model with transfer learning for handwriting recognition."""
    
    def __init__(self, num_classes, device, pretrained=True):
        super(VGG19HandwritingModel, self).__init__()
        self.device = device
        
        # Load a pretrained VGG19 model
        vgg19 = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)

        # For grayscale input, we can either:
        # 1. Use the original VGG with 3-channel input (repeat grayscale channel)
        # 2. Modify first layer for single channel input
        # We'll use approach 1 for simplicity and compatibility with data pipeline
        self.features = vgg19.features

        # Freeze feature parameters if using pretrained model (common practice for transfer learning)
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Replace the classifier with a new one appropriate for num_classes
        # VGG19 with 64x64 input: after 5 maxpools (64 -> 32 -> 16 -> 8 -> 4 -> 2)
        # So, the output from features will be (batch_size, 512, 2, 2)
        num_features_output = 512 * 2 * 2 
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features_output, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # The data pipeline ensures x is (batch, 3, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class VGG19HandwritingModelGrayscale(nn.Module):
    """VGG19-based model modified for single-channel (grayscale) input."""
    
    def __init__(self, num_classes, device, pretrained=True):
        super(VGG19HandwritingModelGrayscale, self).__init__()
        self.device = device
        
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


def get_model(model_name, num_classes, device, pretrained=True):
    """
    Factory function to create models by name.
    
    Args:
        model_name (str): Name of the model ('basic_cnn', 'improved_cnn', 'vgg19', 'vgg19_grayscale')
        num_classes (int): Number of output classes
        device (torch.device): Device to place the model on
        pretrained (bool): Whether to use pretrained weights (for VGG models)
    
    Returns:
        torch.nn.Module: The requested model
    """
    model_name = model_name.lower()
    
    if model_name == 'basic_cnn':
        return LetterCNN64(num_classes).to(device)
    elif model_name == 'improved_cnn':
        return ImprovedLetterCNN(num_classes).to(device)
    elif model_name == 'vgg19':
        return VGG19HandwritingModel(num_classes, device, pretrained).to(device)
    elif model_name == 'vgg19_grayscale':
        return VGG19HandwritingModelGrayscale(num_classes, device, pretrained).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available models: 'basic_cnn', 'improved_cnn', 'vgg19', 'vgg19_grayscale'")


def get_model_info(model_name):
    """
    Get information about a model.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        dict: Information about the model
    """
    info = {
        'basic_cnn': {
            'name': 'Basic CNN',
            'description': 'Simple 3-layer CNN for 64x64 images',
            'parameters': 'Low',
            'training_time': 'Fast',
            'accuracy': 'Good'
        },
        'improved_cnn': {
            'name': 'Improved CNN',
            'description': 'Enhanced CNN with BatchNorm and Dropout',
            'parameters': 'Medium',
            'training_time': 'Medium',
            'accuracy': 'Better'
        },
        'vgg19': {
            'name': 'VGG19 Transfer Learning',
            'description': 'VGG19 with pretrained ImageNet weights',
            'parameters': 'High',
            'training_time': 'Fast (transfer learning)',
            'accuracy': 'Best'
        },
        'vgg19_grayscale': {
            'name': 'VGG19 Grayscale',
            'description': 'VGG19 modified for single-channel input',
            'parameters': 'High',
            'training_time': 'Fast (transfer learning)',
            'accuracy': 'Best'
        }
    }
    
    return info.get(model_name.lower(), {'name': 'Unknown', 'description': 'Model not found'})
