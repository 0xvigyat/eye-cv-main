# Dataset Description

## Overview
This dataset contains retinal images categorized into four classes representing different eye conditions. It is structured for image classification tasks, specifically for diagnosing eye diseases.

## Structure
The dataset is organized into a main folder `Data/dataset/` with four subdirectories, each corresponding to a specific class label:

- **Root Directory:** `Data/dataset/`
- **Classes:**
  - `cataract`
  - `diabetic_retinopathy`
  - `glaucoma`
  - `normal`

## Class Distribution
The dataset consists of a total of **4,217** images distributed across the four classes as follows:

| Class Label | Number of Images | File Format | Description |
| :--- | :--- | :--- | :--- |
| **Cataract** | 1,038 | `.jpg` | Images showing eyes affected by cataract (clouding of the lens). |
| **Diabetic Retinopathy** | 1,098 | `.jpeg` | Images showing signs of diabetic retinopathy (damage to blood vessels in the retina). |
| **Glaucoma** | 1,007 | `.jpg` | Images indicating glaucoma (damage to the optic nerve). |
| **Normal** | 1,074 | `.jpg` | Images of healthy eyes with no visible pathological conditions. |

## Data Balance
The dataset is relatively balanced, with each class containing approximately 1,000 images. This balance is beneficial for training machine learning models as it prevents bias towards a majority class.

- **Largest Class:** Diabetic Retinopathy (1,098 images)
- **Smallest Class:** Glaucoma (1,007 images)
- **Imbalance Ratio:** The ratio between the largest and smallest class is approximately 1.09, indicating a very low imbalance.

## Image Properties
Analysis of the image resolutions and color modes reveals the following characteristics:

| Class | Resolution (Min) | Resolution (Max) | Resolution (Avg) | Common Resolution | Color Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cataract** | 256x256 | 2592x1728 | ~548x467 | 256x256 | RGB |
| **Diabetic Retinopathy** | 512x512 | 512x512 | 512x512 | 512x512 | RGB |
| **Glaucoma** | 256x256 | 2592x1728 | ~561x476 | 256x256 | RGB |
| **Normal** | 512x512 | 512x512 | 512x512 | 512x512 | RGB |

**Key Observations:**
- **Standardized Classes:** The `diabetic_retinopathy` and `normal` classes are perfectly standardized at **512x512** pixels.
- **Variable Classes:** The `cataract` and `glaucoma` classes have varying resolutions, ranging from 256x256 up to high-resolution 2592x1728 images. However, the most common resolution for these is **256x256**.
- **Preprocessing Recommendation:** Due to the resolution variance, it is highly recommended to resize all images to a fixed size (e.g., 224x224 or 256x256) during the data loading/preprocessing pipeline.

## Technical Details
- **Image Formats:** The dataset uses JPEG compression (`.jpg` and `.jpeg`).
- **Color Space:** All images are in **RGB** color mode.
- **Task Suitability:** This dataset is suitable for:
  - Multi-class Image Classification
  - Medical Image Analysis
  - Transfer Learning experiments (e.g., using VGG16, ResNet, EfficientNet)

## Usage
To use this dataset for training a model (e.g., in TensorFlow/Keras or PyTorch), you can use standard data loading utilities like `ImageDataGenerator.flow_from_directory` or `torchvision.datasets.ImageFolder`, pointing them to the `Data/dataset/` directory.
