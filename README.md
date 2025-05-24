# Handwritten Character Recognition Project

## Introduction

This project explores handwritten character recognition using deep learning. It's a journey through building and training different neural network models to identify English alphabet characters (both uppercase and lowercase) and digits. This notebook is designed to be a clear, step-by-step guide, suitable for a high school project, demonstrating how to go from data preparation to model training and finally to predicting characters from images.

## Dataset

The project is designed to work with the "Handwritten English Characters and Digits" dataset, which was originally sourced from Kaggle (though the dataset itself is not included in this repository). This dataset typically consists of images of individual handwritten characters, organized into folders where each folder name corresponds to the character it contains (e.g., 'A', 'b', '7').

The file `content/image_labels.csv` present in the repository was likely used for managing labels for an augmented version of the dataset, where images might have been processed and stored with new filenames. The primary data loading mechanism in the notebook (`HandwritingDataPipeline`) uses PyTorch's `ImageFolder` which automatically infers labels from folder names (e.g., images in a folder named 'A' will be labeled as 'A'). If you are using a dataset structured this way, the CSV might be for reference or for a custom data loading step not implemented in the final notebook.

## Project Structure

*   **`handwriting_recognition.ipynb`**: This is the main Jupyter Notebook. It contains all the Python code for this project, including:
    *   Data loading, preprocessing, and augmentation.
    *   Definitions for several neural network models.
    *   Functions for training and evaluating these models.
    *   Sections for running training experiments.
    *   A dedicated section for performing inference (predicting characters) on new images.
*   **`old_scripts/`**: This folder contains the original Python scripts that were consolidated into the Jupyter Notebook. They are kept for archival purposes.
*   **`content/`**: This directory might contain sample data or the `image_labels.csv` file. The notebook expects the image data to be in a path like `./content/augmented_images/augmented_images1/`.
*   **`model_checkpoints_cnn/`** and **`model_checkpoints_vgg/`**: These folders will be created when you run the training cells in the notebook. They will store the saved model weights (checkpoints).
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `LICENSE`: Contains the license for this project.

## Methodology / What's in the Notebook

The `handwriting_recognition.ipynb` notebook covers the following key steps:

1.  **Setup and Imports:** Initializes the necessary libraries and configures the computing device (CPU/GPU/MPS).
2.  **Data Augmentation:** Defines custom transformations like `ThicknessTransform` to artificially increase the diversity of the training data by simulating variations in writing style (e.g., pen thickness).
3.  **Data Pipeline (`HandwritingDataPipeline`):**
    *   Loads images from the specified directory.
    *   Applies defined augmentations and preprocessing steps (resizing, grayscale conversion, normalization).
    *   Splits the data into training, validation, and test sets.
    *   Creates `DataLoader` instances for efficient batch processing during training and evaluation.
4.  **Model Architectures Explored:**
    *   **`LetterCNN64`**: A basic Convolutional Neural Network (CNN) as an initial attempt.
    *   **`ImprovedLetterCNN`**: An enhanced custom CNN with additions like Batch Normalization (to stabilize learning) and Dropout (to prevent overfitting). This is the primary custom model experimented with.
    *   **`VGG19HandwritingModel`**: Utilizes a pre-trained VGG19 model, a well-known powerful CNN architecture, and adapts it for character recognition. This technique is called transfer learning, leveraging knowledge from a model trained on a large dataset for a new task.
5.  **Training Process:**
    *   Uses Cross-Entropy Loss as the loss function, suitable for multi-class classification.
    *   Employs the Adam optimizer for updating model weights.
    *   Demonstrates experimentation with various learning rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR) which dynamically adjust the learning rate during training to improve convergence and performance.
    *   Saves the best performing model based on validation accuracy during training.
6.  **Evaluation:** Model performance is primarily measured by accuracy on the test set.
7.  **Inference:** A dedicated section in the notebook shows how to:
    *   Load a previously saved trained model.
    *   Prepare a single new image for prediction.
    *   Get the model's prediction (the recognized character) and the confidence of that prediction.

## How to Run

1.  **Setup Environment:**
    *   Ensure you have Python installed.
    *   Install PyTorch and Torchvision: Follow instructions on the [official PyTorch website](https://pytorch.org/).
    *   Install other necessary libraries:
        ```bash
        pip install matplotlib pillow opencv-python numpy jupyterlab albumentations
        ```
        (If you have a `requirements.txt` file, you can often use `pip install -r requirements.txt`)

2.  **Dataset:**
    *   Download or prepare your handwritten character dataset.
    *   Organize it into a directory structure where each sub-directory is named after the character it contains (e.g., `dataset/A/image1.png`, `dataset/B/image2.png`, etc.).
    *   Update the `data_root` variable in the "Data Pipeline Initialization" section of the `handwriting_recognition.ipynb` notebook to point to your dataset's main directory (e.g., if your characters are in `my_data/A`, `my_data/B`, etc., `data_root` would be `my_data/`). The notebook currently points to `"./content/augmented_images/augmented_images1"`.

3.  **Run Jupyter Notebook:**
    *   Open your terminal or command prompt.
    *   Navigate to the project's root directory.
    *   Launch Jupyter Lab:
        ```bash
        jupyter lab
        ```
    *   Open the `handwriting_recognition.ipynb` file from the Jupyter Lab interface.
    *   You can run cells individually or run all cells.
    *   For inference, navigate to the "Inference on a Single Image" section and follow the instructions to provide an image path.

## Results/Observations (Conceptual)

The notebook is structured to allow you to train different models and observe their performance. By running the training cells, you'll see how the training and validation loss/accuracy change over epochs. The test accuracy will give you a final measure of how well the model generalizes to unseen data. You can then compare the custom CNN approaches with the VGG19 transfer learning approach. The inference section allows for practical application of your trained models.

Happy character recognizing!
