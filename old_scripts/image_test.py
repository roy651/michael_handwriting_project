import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from image_class import ImprovedLetterCNN

def load_model(model_path, num_classes, device):
    """Load the trained model."""
    model = ImprovedLetterCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def prepare_image(image_path):
    """Prepare image for classification."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def get_class_labels(data_root):
    """Get class labels from the data directory."""
    class_names = sorted(os.listdir(data_root))
    # Remove any hidden files (like .DS_Store)
    class_names = [c for c in class_names if not c.startswith('.')]
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Classify a single handwritten character image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model_path', default='checkpoints/best_model.pth', help='Path to the trained model')
    parser.add_argument('--data_root', required=True, help='Path to training data directory (for class labels)')
    args = parser.parse_args()

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get class labels and number of classes
    class_names = get_class_labels(args.data_root)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Load model
    try:
        model = load_model(args.model_path, num_classes, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load and prepare image
    try:
        image = prepare_image(args.image_path)
        image = image.to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    # Print results
    print(f"\nClassification Results:")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

    # Print top 3 predictions
    top_k = 3
    top_probs, top_classes = torch.topk(probabilities, top_k)
    print(f"\nTop {top_k} predictions:")
    for i in range(top_k):
        class_idx = top_classes[0][i].item()
        prob = top_probs[0][i].item() * 100
        print(f"{class_names[class_idx]}: {prob:.2f}%")

if __name__ == "__main__":
    main()