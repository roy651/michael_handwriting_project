# Handwritten Character Recognition Project

## 1. Overview

This project implements a handwritten character recognition system using PyTorch. It includes functionalities for data preprocessing and augmentation, model training (custom CNNs and transfer learning with VGG19), inference on single images and character sequences, and an interactive web application built with Gradio for real-time recognition.

The system is designed to recognize individual English alphabet characters (uppercase and lowercase) and digits (0-9).

## 2. Features

- **Data Pipeline**: Robust data loading, preprocessing, augmentation, and splitting (training, validation, test sets).
- **Augmentation**: Includes techniques like random affine transformations, perspective shifts, rotation, thickness variation, and Gaussian blur to improve model generalization.
- **Model Architectures**:
  - `LetterCNN64`: A basic custom CNN for 64x64 pixel inputs.
  - `ImprovedLetterCNN`: An enhanced custom CNN with Batch Normalization and Dropout.
  - `VGG19HandwritingModel`: A transfer learning model using a pre-trained VGG19 architecture.
- **Training**: Comprehensive training script with learning rate scheduling, checkpointing (saving best and final models), and performance logging (loss/accuracy).
- **Inference**:
  - Scripts for predicting single characters from image files.
  - Detailed character extraction and classification from images containing multiple characters, with visualization of intermediate steps.
- **Interactive Application**: A Gradio web UI (`gradio.ipynb`) allowing:
  - Model selection (between trained CNN and VGG models).
  - Multiple input methods: drawing (single character), image upload (single/multiple characters), and webcam capture (multiple characters).
  - Visualization of processing steps: original image, grayscale, binarized versions, contour detection, segmented ROIs, and final predictions with bounding boxes.

## 3. Notebook Descriptions

The project is organized into several Jupyter notebooks:

- **`data.ipynb`**:

  - Handles dataset loading, exploration, and visualization.
  - Defines and showcases data augmentation techniques.
  - Includes the `HandwritingDataPipeline` for preparing data for model training.
  - Performs basic dataset analysis, such as class distribution.

- **`training.ipynb`**:

  - Defines the model architectures (`LetterCNN64`, `ImprovedLetterCNN`, `VGG19HandwritingModel`).
  - Contains the core `train_model` function for training loops, checkpointing, and evaluation.
  - Includes utilities like `load_model` (for resuming training or loading for tests) and `test_model`.
  - Conducts training experiments for the defined models and saves checkpoints to `./model_checkpoints/`.

- **`inference.ipynb`**:

  - Focuses on using trained models for prediction.
  - Includes `load_model_for_inference` to load saved model weights.
  - Provides `prepare_image_for_inference` for processing single images.
  - Demonstrates inference on a single sample image.
  - Features `extract_letters_detailed_visualization` for segmenting and classifying characters in an image containing text, saving intermediate visualizations.

- **`gradio.ipynb`**:

  - The main interactive application notebook.
  - Loads trained models (CNN and VGG) from checkpoints.
  - Provides a user-friendly Gradio interface with tabs for:
    - Drawing a single character for recognition.
    - Uploading an image of characters.
    - Capturing characters via webcam.
  - Displays recognized text and a gallery of processing steps.

- **`app.ipynb`**:

  - This notebook might contain earlier or alternative versions of the Gradio application. For the most up-to-date and comprehensive Gradio demo, please use `gradio.ipynb`.

- **`old_scripts/` (Directory)**:
  - Contains previous versions of notebooks (`handwriting_recognition.ipynb`, `handwriting_training.ipynb`, `handwriting_inference.ipynb`, `temp_consolidated.ipynb`) and Python scripts developed during the project. This directory serves as an archive of the project's evolution.

## 4. Setup and Installation

1.  **Clone the repository:**

        ```bash
        git clone <repository_url>
        cd <repository_directory>
        ```

2.  **Create a Python virtual environment (recommended):**
    `bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    `

3.  **Install dependencies:**
    The primary dependencies are PyTorch, Torchvision, OpenCV, PIL (Pillow), Matplotlib, NumPy, and Gradio.
    A `pyproject.toml` file is provided, which can be used with tools like Poetry or modern pip.
    If using pip with `uv` (as suggested by `uv.lock`):

    ```bash
    pip install uv
    uv pip install -r requirements.txt  # Assuming you generate a requirements.txt
    ```

    Alternatively, install manually (ensure you get the correct PyTorch version for your system from [pytorch.org](https://pytorch.org/)):

    ```bash
    pip install torch torchvision torchaudio
    pip install opencv-python-headless pillow matplotlib numpy gradio==3.50.2
    ```

    _Note: `gradio==3.50.2` is specified in `gradio.ipynb` for stability._

## 5. Usage

### 5.1. Data Exploration and Preparation (`data.ipynb`)

Open and run the cells in `data.ipynb` to:

- Visualize sample images and the effects of various augmentations.
- See the class distribution of your dataset.
- Understand how the `HandwritingDataPipeline` works.

### 5.2. Model Training (`training.ipynb`)

Open and run the cells in `training.ipynb` to:

- Define model architectures.
- Train the `ImprovedLetterCNN` and/or `VGG19HandwritingModel`.
- Model checkpoints will be saved to `./model_checkpoints/cnn/` and `./model_checkpoints/vgg/` respectively.
- Training progress, validation accuracy, and loss plots will be displayed.
- **Important**: Ensure `data_root_example` in this notebook points to your dataset.

### 5.3. Inference (`inference.ipynb`)

Open and run the cells in `inference.ipynb`:

- **Single Image Inference**:
  - Update `user_model_checkpoint_path` to your chosen trained model (e.g., `./model_checkpoints/cnn/best_model.pth`).
  - Update `user_data_root_for_labels` to your dataset path for class label mapping.
  - Update `user_image_for_prediction` with the path to an image you want to classify.
- **Detailed Multi-Character Analysis**:
  - Update `user_multi_char_image_path` to an image containing multiple characters.
  - This section uses the model loaded in the single image inference part.
  - Intermediate processing steps will be saved in `./detailed_extraction_output/`.

### 5.4. Launching the Gradio Web Application (`gradio.ipynb`)

Open and run the cells in `gradio.ipynb`:

- Ensure `DATA_ROOT_FOR_LABELS` is correctly set to your dataset path.
- Ensure `MODEL_CKPT_DIR_CNN` and `MODEL_CKPT_DIR_VGG` point to the directories where your trained models (`best_model.pth`) are saved (typically `./model_checkpoints/cnn/` and `./model_checkpoints/vgg/`).
- The notebook will load available models and class labels.
- The final cell will launch the Gradio application. You can interact with it through your web browser using the provided local (and potentially public, if `share=True` is used and you are in a suitable environment like Colab) URL.
- **Features of the Gradio App**:
  - Select the model (CNN or VGG) to use for recognition.
  - **Draw Single Character Tab**: Draw a character in the sketchpad and get a prediction.
  - **Upload Multi-Character Image Tab**: Upload an image containing one or more characters. The app will attempt to segment and recognize them.
  - **Webcam Capture Tab**: Use your webcam to capture an image of characters for recognition.
  - View recognized text and a gallery of processing visualizations.

## 6. Model Checkpoints

- Trained model checkpoints are saved during the execution of `training.ipynb`.
- Default save locations:
  - Custom CNN models: `./model_checkpoints/cnn/`
  - VGG19 models: `./model_checkpoints/vgg/`
- Both `best_model.pth` (based on validation accuracy) and `final_model.pth` (after all epochs) are saved. The Gradio app and inference notebook typically use `best_model.pth`.

## 7. Potential Future Work/Improvements

- **Advanced Segmentation**: Implement more sophisticated text segmentation algorithms for better handling of connected or overlapping characters.
- **Expanded Character Set**: Train on datasets with a wider range of symbols, special characters, or different languages.
- **Real-time Webcam Stream Processing**: Enhance the Gradio app to process webcam input as a continuous stream rather than single captures.
- **Hyperparameter Optimization**: Use techniques like Optuna or Ray Tune for more systematic hyperparameter tuning.
- **Deployment**: Package the application for deployment using tools like Docker and host on cloud platforms.
- **Attention Mechanisms**: Explore adding attention mechanisms to the models for potentially improved focus on relevant character features.

## 8. Contributing

Contributions to this project are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes, ensuring code is well-commented and tested.
4.  Submit a pull request with a clear description of your changes.
