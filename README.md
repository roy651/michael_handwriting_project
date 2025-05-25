# Handwritten Character Recognition Project

## 1. Overview

This project implements a comprehensive handwritten character recognition system using PyTorch. It recognizes individual English alphabet characters (uppercase and lowercase) and digits (0-9) through a pipeline of data preprocessing, model training, and inference. The system features both custom CNN architectures and transfer learning with VGG19, along with an interactive Gradio web interface for real-time recognition.

## 2. Features

- **Data Pipeline**: Robust data loading, preprocessing, augmentation, and dataset splitting
- **Augmentation Techniques**: Random affine transformations, perspective shifts, rotation, thickness variation, and Gaussian blur
- **Model Architectures**:
  - `LetterCNN64`: Basic custom CNN for 64x64 pixel inputs
  - `ImprovedLetterCNN`: Enhanced CNN with Batch Normalization and Dropout
  - `VGG19HandwritingModel`: Transfer learning model using pre-trained VGG19
- **Training**: Learning rate scheduling, checkpointing, and performance logging
- **Inference**: Single character prediction and multi-character extraction/classification
- **Interactive Application**: Gradio web UI with multiple input methods and visualization tools

## 3. Project Structure

The project is organized into the following components:

### 3.1. Notebooks

- **`data.ipynb`**: Dataset exploration, visualization, and augmentation techniques
- **`training.ipynb`**: Model definition, training pipeline, and experimentation
- **`inference.ipynb`**: Model loading and inference on single/multiple characters
- **`gradio.ipynb`**: Interactive web application for real-time character recognition

### 3.2. Utility Files

- **`data_utils_file.py`**: Data loading and augmentation utilities
- **`models_file.py`**: Model architecture definitions
- **`training_utils_file.py`**: Training loop and optimization utilities
- **`inference_utils_file.py`**: Inference and visualization functions

### 3.3. Directories

- **`model_checkpoints/`**: Saved model weights (created during training)
- **`example_images/`**: Sample images for demo purposes (created by Gradio if needed)
- **`inference_results/`**: Output visualizations from inference (created during inference)
- **`gradio_results/`**: Results from Gradio interface (created during Gradio usage)

## 4. Setup and Installation

### 4.1. Clone the Repository

```bash
git clone https://github.com/roy651/michael_handwriting_project.git
cd michael_handwriting_project
```

### 4.2. Create a Virtual Environment

It's recommended to use a virtual environment to avoid package conflicts:

```bash
# Using venv (Python's built-in module)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR using conda
conda create -n handwriting_env python=3.9
conda activate handwriting_env
```

### 4.3. Install Dependencies

#### Option 1: Using requirements.txt

```bash
pip install -r requirements.txt
```

#### Option 2: Manual installation

```bash
# Install PyTorch (visit pytorch.org for the appropriate command for your system)
pip install torch torchvision

# Install other dependencies
pip install numpy matplotlib opencv-python pillow
pip install gradio==3.36.1  # Use this specific version for compatibility
pip install albumentations
```

### 4.4. Verify Installation

Run the following in a Python console to verify the key packages are installed:

```python
import torch
import cv2
import numpy as np
import gradio
import matplotlib.pyplot as plt
from PIL import Image

print(f"PyTorch version: {torch.__version__}")
print(f"Gradio version: {gradio.__version__}")
print(f"OpenCV version: {cv2.__version__}")
```

## 5. Usage Guide

### 5.1. Data Analysis and Exploration (`data.ipynb`)

This notebook helps you understand your dataset and explore data augmentation techniques.

1. **Run the notebook**:

   ```bash
   jupyter notebook data.ipynb
   ```

2. **Key functions**:

   - `load_and_explore_dataset()`: Examine dataset structure and class distribution
   - `visualize_sample_images()`: View examples from the dataset
   - `analyze_image_properties()`: Analyze dimensions, aspect ratios, and pixel intensities
   - `extract_stroke_features()`: Examine character stroke thickness and continuity
   - `visualize_augmentation_techniques()`: See the effects of different augmentations

3. **Setup required**:
   - Update the `DATA_ROOT` variable to point to your dataset directory
   - Ensure the dataset is organized with class subdirectories (e.g., `/0`, `/1`, `/A`, `/B`)

### 5.2. Model Training (`training.ipynb`)

This notebook defines model architectures and trains them on your dataset.

1. **Run the notebook**:

   ```bash
   jupyter notebook training.ipynb
   ```

2. **Key functions**:

   - `setup_data_pipeline()`: Configure data loading and preprocessing
   - `train_and_evaluate_model()`: Train a single model with specified parameters
   - `train_multiple_models()`: Compare different model architectures
   - `fine_tune_model()`: Two-phase training for transfer learning models
   - `run_complete_training_pipeline()`: End-to-end training workflow

3. **Setup required**:

   - Update the `DATA_ROOT` variable to point to your dataset
   - Adjust training parameters (batch size, learning rate, etc.) as needed
   - Models will be saved to `model_checkpoints/` directory

4. **Training examples**:

   ```python
   # Train a basic CNN model
   model, history = train_and_evaluate_model(
       model_name='improved_cnn',
       train_loader=train_loader,
       val_loader=val_loader,
       test_loader=test_loader,
       num_classes=len(class_names),
       num_epochs=20
   )

   # OR use the complete pipeline
   trained_model, history, class_names = run_complete_training_pipeline(
       data_root=DATA_ROOT,
       model_name='improved_cnn',
       num_epochs=20
   )
   ```

### 5.3. Inference (`inference.ipynb`)

This notebook demonstrates how to use trained models for predictions.

1. **Run the notebook**:

   ```bash
   jupyter notebook inference.ipynb
   ```

2. **Key functions**:

   - `load_model_checkpoint()`: Load a trained model from a checkpoint file
   - `infer_single_image()`: Predict the class of a single character image
   - `recognize_handwritten_text()`: Extract and recognize characters from text images
   - `visualize_model_confidence()`: Examine model's confidence across classes
   - `benchmark_model_performance()`: Measure inference speed

3. **Setup required**:

   - Update model checkpoint paths to point to your saved models
   - Prepare test images for inference
   - Ensure class names match those used during training

4. **Inference examples**:

   ```python
   # Single image inference
   result = infer_single_image(
       model=model,
       class_names=class_names,
       image_path="path/to/your/image.jpg"
   )

   # Text recognition
   text, char_info, visualizations = recognize_handwritten_text(
       model=model,
       class_names=class_names,
       image_path="path/to/text_image.jpg"
   )
   ```

### 5.4. Gradio Interactive App (`gradio.ipynb`)

This notebook provides an interactive web interface for real-time character recognition.

1. **Run the notebook**:

   ```bash
   jupyter notebook gradio.ipynb
   ```

2. **Setup required**:

   - Ensure your trained model checkpoints are available
   - Models should be organized in a structure like:
     ```
     model_checkpoints/
     ├── model_name1/
     │   ├── best_model.pth
     │   └── class_names.txt (optional)
     ├── model_name2/
     │   └── best_model.pth
     ```

3. **Usage guide**:

   - The app will automatically scan for available models in the `model_checkpoints/` directory
   - Select a model from the dropdown menu
   - Use one of the input methods (Upload, Draw, or Camera) to provide an image
   - Click "Recognize Characters" to process the image
   - View the results showing detected characters and extracted segments

4. **Troubleshooting**:

   - If you encounter errors about missing attributes, check your Gradio version
   - For compatibility with older Gradio versions (3.36.x), modify the `create_examples_section()` function as described in the notebook comments
   - If models aren't detected, verify your checkpoint directory structure
   - For image processing issues, adjust the settings in the Recognition Settings accordion

5. **Running the minimal interface**:
   If you encounter issues with the full interface, the notebook also provides a simplified version:
   ```python
   # Launch the simplified interface for more reliable operation
   launch_simple_application(share=True)
   ```

## 6. Working with Trained Models

### 6.1. Model Checkpoint Structure

Trained models are saved with the following information:

- Model state dictionary (weights and biases)
- Optimizer state
- Training epoch
- Validation accuracy
- Learning rate scheduler state (if used)

### 6.2. Using Pre-trained Models

To use a pre-trained model directly with Gradio:

1. Ensure your checkpoints are organized in the required directory structure
2. Run the Gradio notebook to automatically detect and load models
3. If class names aren't available, the system will use default labels (0-9, A-Z, a-z)

### 6.3. Converting Models for Deployment

To export models for deployment:

```python
# Load and export the model
import torch
from models_file import get_model

# Initialize model architecture
model = get_model('improved_cnn', num_classes=62, device='cpu')

# Load weights
checkpoint = torch.load('model_checkpoints/cnn/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to TorchScript (optional)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'deployment/model.pt')
```

## 7. Gradio App: Tips and Features

The Gradio application provides a user-friendly interface for handwritten character recognition with several useful features:

### 7.1. Input Methods

- **Upload Image**: Upload an image file containing handwritten characters
- **Draw Text**: Use the sketch pad to draw characters directly in the browser
- **Camera Capture**: Take a photo using your device's camera

### 7.2. Recognition Settings

The app provides customization options through the "Recognition Settings" accordion:

- **Checkpoint Type**: Choose between "best" or "final" model checkpoints
- **Minimum Character Area**: Adjust this value to control character segmentation sensitivity
- **Image Enhancement**: Apply preprocessing techniques to improve recognition
- **Character Spacing**: Control the padding around extracted characters

### 7.3. Debug & Visualization

The "Debug & Visualization" accordion provides:

- **Show Processing Steps**: Toggle visualization of intermediate processing steps
- **Show Confidence Scores**: Display confidence values for each prediction
- **Debug Information**: View detailed logs of the recognition process

### 7.4. Batch Processing

For processing multiple images at once:

1. Use the "Batch Processing" tab
2. Upload multiple image files
3. Click "Process Batch"
4. View the gallery of results and download a CSV summary

### 7.5. Troubleshooting

- **Character Segmentation Issues**: Adjust the "Minimum Character Area" slider
- **Poor Recognition Quality**: Try different "Image Enhancement" options
- **Model Not Found**: Verify checkpoint directory structure
- **Interface Errors**: Check Gradio version compatibility (recommended: 3.36.1)

## 8. Future Work and Improvements

- **Advanced Segmentation**: Implement better text segmentation for connected characters
- **Expanded Character Set**: Add support for more symbols and languages
- **Real-time Webcam Processing**: Enable continuous recognition from video stream
- **Hyperparameter Optimization**: Apply systematic tuning for improved accuracy
- **Mobile Deployment**: Create lightweight models for mobile applications
- **OCR Integration**: Combine with text line detection for full document OCR

## 9. Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 10. License

[Add your license information here]

## 11. Acknowledgments

- Dataset source: [Add dataset attribution if applicable]
- PyTorch team for the deep learning framework
- Gradio team for the interactive UI framework
