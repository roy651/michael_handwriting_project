"""
Inference utilities for handwritten character recognition.
Contains functions for single image inference, batch processing, and character extraction.
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms


def prepare_single_image(image_path, image_size=(64, 64), normalization_type='imagenet', 
                        device='cpu', grayscale_input=False):
    """
    Prepare a single image for model inference.
    
    Args:
        image_path (str): Path to the image file
        image_size (tuple): Target image size (height, width)
        normalization_type (str): Type of normalization ('imagenet', 'simple', 'grayscale')
        device (str): Device to load tensor on
        grayscale_input (bool): Whether model expects single channel input
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found")
    
    # Setup normalization based on type
    if normalization_type == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        channels = 3
    elif normalization_type == 'simple':
        normalize = transforms.Normalize((0.5,), (0.5,))
        channels = 1
    elif normalization_type == 'grayscale':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        channels = 3
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")
    
    # Build transform pipeline
    transform_list = [
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
    
    # Add channel adjustment if needed
    if not grayscale_input and channels == 3:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
    
    transform_list.append(normalize)
    transform = transforms.Compose(transform_list)
    
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise RuntimeError(f"Error processing image '{image_path}': {str(e)}")


def prepare_pil_image(pil_image, image_size=(64, 64), normalization_type='imagenet', 
                     device='cpu', grayscale_input=False):
    """
    Prepare a PIL image for model inference.
    
    Args:
        pil_image: PIL Image object
        image_size (tuple): Target image size (height, width)
        normalization_type (str): Type of normalization
        device (str): Device to load tensor on
        grayscale_input (bool): Whether model expects single channel input
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
    """
    # Setup normalization based on type
    if normalization_type == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        channels = 3
    elif normalization_type == 'simple':
        normalize = transforms.Normalize((0.5,), (0.5,))
        channels = 1
    elif normalization_type == 'grayscale':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        channels = 3
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")
    
    # Build transform pipeline
    transform_list = [
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
    
    # Add channel adjustment if needed
    if not grayscale_input and channels == 3:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))
    
    transform_list.append(normalize)
    transform = transforms.Compose(transform_list)
    
    try:
        image_tensor = transform(pil_image)
        return image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise RuntimeError(f"Error processing PIL image: {str(e)}")


def predict_single_image(model, image_path, class_names, device='cpu', 
                        normalization_type='imagenet', return_probabilities=False):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained PyTorch model
        image_path (str): Path to image file
        class_names (list): List of class names
        device (str): Device to run inference on
        normalization_type (str): Type of normalization
        return_probabilities (bool): Whether to return all class probabilities
        
    Returns:
        dict: Prediction results containing class, confidence, and optionally probabilities
    """
    model.eval()
    model = model.to(device)
    
    # Prepare image
    image_tensor = prepare_single_image(image_path, normalization_type=normalization_type, device=device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'predicted_index': predicted_idx.item()
        }
        
        if return_probabilities:
            class_probs = {class_names[i]: probabilities[0][i].item() 
                          for i in range(len(class_names))}
            result['class_probabilities'] = class_probs
        
        return result


def predict_batch_images(model, image_paths, class_names, device='cpu', 
                        normalization_type='imagenet', batch_size=32):
    """
    Predict classes for a batch of images.
    
    Args:
        model: Trained PyTorch model
        image_paths (list): List of image file paths
        class_names (list): List of class names
        device (str): Device to run inference on
        normalization_type (str): Type of normalization
        batch_size (int): Batch size for processing
        
    Returns:
        list: List of prediction results for each image
    """
    model.eval()
    model = model.to(device)
    
    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        
        # Prepare batch of images
        for path in batch_paths:
            try:
                tensor = prepare_single_image(path, normalization_type=normalization_type, device=device)
                batch_tensors.append(tensor.squeeze(0))  # Remove batch dimension for stacking
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({'error': str(e), 'image_path': path})
                continue
        
        if not batch_tensors:
            continue
            
        # Stack tensors and run inference
        batch_tensor = torch.stack(batch_tensors)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)
            
            # Process results
            for j, path in enumerate(batch_paths):
                if j < len(predicted_indices):  # Ensure we have a prediction
                    predicted_class = class_names[predicted_indices[j].item()]
                    confidence_score = confidences[j].item()
                    
                    results.append({
                        'image_path': path,
                        'predicted_class': predicted_class,
                        'confidence': confidence_score,
                        'predicted_index': predicted_indices[j].item()
                    })
    
    return results


def extract_characters_from_image(image_path, model, class_names, device='cpu',
                                 spacing=24, min_area=50, normalization_type='simple',
                                 visualize=False, output_dir=None):
    """
    Extract and classify individual characters from a handwritten text image.
    
    Args:
        image_path (str): Path to the input image
        model: Trained model for character classification
        class_names (list): List of class names
        device (str): Device to run inference on
        spacing (int): Spacing around characters when preprocessing
        min_area (int): Minimum contour area to consider as a character
        normalization_type (str): Type of normalization for model input
        visualize (bool): Whether to create visualization images
        output_dir (str): Directory to save extracted letters (optional)
        
    Returns:
        tuple: (detected_text, character_info_list, visualization_images)
    """
    model.eval()
    model = model.to(device)
    
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Read and process the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    visualization_images = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if visualize:
        visualization_images.append(("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        visualization_images.append(("Grayscale Image", gray))
    
    # Apply thresholding for contour detection (inverted for finding text)
    _, binary_contours = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply thresholding for model input (non-inverted)
    _, binary_model = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if visualize:
        visualization_images.append(("Binary for Contours", binary_contours))
        visualization_images.append(("Binary for Model", binary_model))
    
    # Find contours
    contours, _ = cv2.findContours(binary_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    letter_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if visualize:
        contour_img = image.copy()
        cv2.drawContours(contour_img, letter_contours, -1, (0, 255, 0), 2)
        visualization_images.append(("Filtered Contours", cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)))
    
    # Sort contours from left to right (reading order)
    letter_contours = sorted(letter_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    
    # Process each character
    character_info = []
    detected_text = ""
    
    for i, contour in enumerate(letter_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract character from binary image
        char_image = binary_model[y:y+h, x:x+w]
        
        # Save original extracted character if output directory provided
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"char_{i}_original.png"), char_image)
        
        # Process character for model input
        processed_char = _process_character_for_model(char_image, spacing)
        
        # Save processed character if output directory provided
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"char_{i}_processed.png"), processed_char)
        
        # Convert to PIL and then to tensor
        char_pil = Image.fromarray(processed_char)
        char_tensor = prepare_pil_image(char_pil, normalization_type=normalization_type, 
                                      device=device, grayscale_input=(normalization_type == 'simple'))
        
        # Classify character
        with torch.no_grad():
            outputs = model(char_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_char = class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            character_info.append({
                'character': predicted_char,
                'confidence': confidence_score,
                'bbox': (x, y, w, h),
                'index': i
            })
            
            detected_text += predicted_char
    
    # Create final visualization with bounding boxes and predictions
    if visualize and character_info:
        result_img = image.copy()
        for info in character_info:
            x, y, w, h = info['bbox']
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, f"{info['character']} ({info['confidence']:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        visualization_images.append(("Final Result", cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))
    
    return detected_text, character_info, visualization_images


def _process_character_for_model(char_image, spacing=24):
    """
    Process extracted character image for model input.
    
    Args:
        char_image: Character image as numpy array
        spacing: Spacing around character
        
    Returns:
        numpy.ndarray: Processed character image
    """
    # Convert to PIL for easier manipulation
    char_pil = Image.fromarray(char_image)
    h, w = char_image.shape
    
    # Calculate target size while maintaining aspect ratio
    target_size = 64 - spacing
    
    if w > h:
        new_width = target_size
        new_height = int((h / w) * target_size)
    else:
        new_height = target_size
        new_width = int((w / h) * target_size)
    
    # Resize while maintaining aspect ratio
    resized_image = char_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # Create padded image
    padded_image = Image.new('L', (target_size, target_size), 255)
    
    # Calculate padding to center the image
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2
    
    # Paste the resized image at the center
    padded_image.paste(resized_image, (pad_x, pad_y))
    
    # Create final 64x64 image with spacing
    final_image = Image.new('L', (64, 64), 255)
    spacing_padding = spacing // 2
    final_image.paste(padded_image, (spacing_padding, spacing_padding))
    
    return np.array(final_image)


def process_numpy_image(image_array, model, class_names, device='cpu', 
                       normalization_type='simple', min_area=50):
    """
    Process a numpy image array (e.g., from Gradio) for character extraction.
    
    Args:
        image_array: Numpy array representing the image
        model: Trained model for character classification
        class_names: List of class names
        device: Device to run inference on
        normalization_type: Type of normalization
        min_area: Minimum contour area
        
    Returns:
        tuple: (detected_text, visualization_images)
    """
    model.eval()
    model = model.to(device)
    
    # Convert RGB to BGR if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image = image_array
    
    visualization_images = []
    
    # Original image
    orig_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    visualization_images.append(("Original Image", orig_img))
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    visualization_images.append(("Grayscale Image", gray))
    
    # Thresholding
    _, binary_contours = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary_model = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    visualization_images.append(("Binary for Contours", binary_contours))
    visualization_images.append(("Binary for Model", binary_model))
    
    # Find and filter contours
    contours, _ = cv2.findContours(binary_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    letter_contours = sorted(letter_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    
    # Visualize contours
    if len(image.shape) == 3:
        contour_img = image.copy()
        cv2.drawContours(contour_img, letter_contours, -1, (0, 255, 0), 2)
        visualization_images.append(("Detected Contours", cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)))
    
    # Process characters
    detected_text = ""
    character_info = []
    
    for i, contour in enumerate(letter_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract and process character
        char_image = binary_model[y:y+h, x:x+w]
        processed_char = _process_character_for_model(char_image)
        
        # Convert to tensor and classify
        char_pil = Image.fromarray(processed_char)
        char_tensor = prepare_pil_image(char_pil, normalization_type=normalization_type, 
                                      device=device, grayscale_input=(normalization_type == 'simple'))
        
        with torch.no_grad():
            outputs = model(char_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_char = class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            character_info.append({
                'character': predicted_char,
                'confidence': confidence_score,
                'bbox': (x, y, w, h)
            })
            
            detected_text += predicted_char
    
    # Final result visualization
    if len(image.shape) == 3:
        result_img = image.copy()
        for info in character_info:
            x, y, w, h = info['bbox']
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, f"{info['character']} ({info['confidence']:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        visualization_images.append(("Final Result", cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))
    
    return detected_text, visualization_images


def create_visualization_grid(images_with_titles, grid_cols=3, figsize=(15, 10), save_path=None):
    """
    Create a grid visualization of multiple images.
    
    Args:
        images_with_titles: List of tuples (title, image)
        grid_cols: Number of columns in the grid
        figsize: Figure size
        save_path: Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    num_images = len(images_with_titles)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    
    # Handle single row case
    if grid_rows == 1:
        axes = [axes] if grid_cols == 1 else axes
    elif grid_cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, (title, image) in enumerate(images_with_titles):
        row = i // grid_cols
        col = i % grid_cols
        
        if grid_rows == 1:
            ax = axes[col] if grid_cols > 1 else axes
        else:
            ax = axes[row][col] if grid_cols > 1 else axes[row]
        
        # Display image
        if len(image.shape) == 2:  # Grayscale
            ax.imshow(image, cmap='gray')
        else:  # RGB
            ax.imshow(image)
        
        ax.set_title(title)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_images, grid_rows * grid_cols):
        row = i // grid_cols
        col = i % grid_cols
        
        if grid_rows == 1:
            ax = axes[col] if grid_cols > 1 else axes
        else:
            ax = axes[row][col] if grid_cols > 1 else axes[row]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def get_top_k_predictions(model, image_tensor, class_names, k=5, device='cpu'):
    """
    Get top-k predictions for an image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
        k: Number of top predictions to return
        device: Device to run inference on
        
    Returns:
        list: List of dictionaries with class names and confidence scores
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k)
        
        predictions = []
        for i in range(k):
            predictions.append({
                'class': class_names[top_k_indices[0][i].item()],
                'confidence': top_k_probs[0][i].item(),
                'index': top_k_indices[0][i].item()
            })
        
        return predictions


def benchmark_inference_speed(model, sample_image_path, class_names, device='cpu', 
                             num_iterations=100, warmup_iterations=10):
    """
    Benchmark inference speed of the model.
    
    Args:
        model: Trained model
        sample_image_path: Path to sample image for benchmarking
        class_names: List of class names
        device: Device to run benchmark on
        num_iterations: Number of inference iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict: Benchmark results
    """
    model.eval()
    model = model.to(device)
    
    # Prepare sample image
    image_tensor = prepare_single_image(sample_image_path, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(image_tensor)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model(image_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_iterations
    fps = 1.0 / avg_time_per_inference
    
    return {
        'total_time': total_time,
        'avg_time_per_inference': avg_time_per_inference,
        'fps': fps,
        'num_iterations': num_iterations,
        'device': device
    }
