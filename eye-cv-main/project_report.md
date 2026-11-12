# Retinal Disease Classification Project Report

## 1. Executive Summary
This project aimed to develop a robust deep learning system for the automated classification of retinal diseases from fundus images. We successfully implemented and evaluated three state-of-the-art architectures: **ResNet50**, **EfficientNetB0**, and **Vision Transformer (ViT-B16)**.

Our experiments demonstrate that **ResNet50** and **ViT-B16** are the top-performing models, both achieving **~95% validation accuracy**. This high level of performance suggests that the system is ready for potential clinical decision support applications.

## 2. Dataset & Preprocessing

### 2.1 Dataset
The dataset consists of retinal fundus images categorized into four classes:
1.  **Cataract**
2.  **Diabetic Retinopathy**
3.  **Glaucoma**
4.  **Normal**

### 2.2 Preprocessing Pipeline
To ensure model robustness and handle variations in lighting and contrast, we implemented a specialized medical image preprocessing pipeline:
*   **Color Space Conversion:** Images were converted to the **LAB color space**.
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied to the L-channel (Luminance) to enhance local contrast while preventing noise amplification. This is crucial for revealing subtle features like exudates and microaneurysms.
*   **Resizing:** All images were resized to **224x224** pixels to match the input requirements of the pre-trained models.

## 3. Methodology

We employed a **Transfer Learning** approach, leveraging models pre-trained on ImageNet to extract rich feature representations.

### 3.1 Architectures
1.  **ResNet50:** A deep Convolutional Neural Network (CNN) with residual connections. Known for its ability to train very deep networks without vanishing gradient issues.
2.  **EfficientNetB0:** A highly efficient CNN that scales depth, width, and resolution uniformly. It offers a good balance between performance and computational cost.
3.  **Vision Transformer (ViT-B16):** A non-convolutional architecture based on the Transformer mechanism. It splits images into patches and processes them using self-attention, allowing it to capture long-range dependencies across the image.

### 3.2 Training Configuration
*   **Optimizer:** Adam
*   **Learning Rate:** 1e-4
*   **Loss Function:** Categorical Crossentropy
*   **Epochs:** 30
*   **Batch Size:** 32
*   **Data Augmentation:** Random rotation, zoom, and horizontal flip were used to prevent overfitting.

## 4. Experimental Results

We evaluated the models based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

### 4.1 Performance Comparison

| Model | Validation Accuracy | Best Class (Recall) | Worst Class (Recall) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **ViT-B16** | **94%** | Diabetic Retinopathy (95%) | Normal (87%) | **Incredible improvement**. Resolved previous mode collapse. Excellent precision/recall across all classes. |
| **ResNet50** | **95%** | Normal (96%) | Glaucoma (93%) | Extremely robust and balanced. Slightly better at "Normal" cases than ViT. |
| **EfficientNetB0** | 87% | Cataract (93%) | Diabetic Retinopathy (81%) | Strong performance, but slightly lagged behind the other two in distinguishing DR and Glaucoma. |

### 4.2 Analysis
*   **ResNet50** proved to be the most reliable "out-of-the-box" performer, showing a very balanced confusion matrix and high recall across all classes.
*   **ViT-B16**, initially struggling with mode collapse, made a massive comeback after hyperparameter tuning. Its ability to achieve **95% recall on Diabetic Retinopathy** is particularly impressive, as DR features can be subtle and scattered.
*   **EfficientNetB0** performed well but struggled slightly with the fine-grained distinctions required for early-stage disease detection compared to the larger models.

## 5. Conclusion

This project successfully demonstrated that deep learning models can achieve expert-level performance in classifying retinal diseases.

*   **Recommendation:** For deployment, we recommend **ResNet50** for its slight edge in overall stability and computational efficiency on standard hardware. However, **ViT-B16** remains a powerful alternative, especially if interpretability via attention maps is a priority.
*   **Future Work:**
    *   **Ensembling:** Combining predictions from ResNet50 and ViT could potentially push accuracy even higher.
    *   **Explainability:** Further refining the Grad-CAM and Attention Map visualizations to provide clinicians with interpretable "heatmaps" of disease regions.

## 6. Project Artifacts
*   `src/`: Source code for data loading, model definition, and training.
*   `models/`: Saved weights for the trained models.
*   `colab_explainability.py`: Script for generating model explanations (Grad-CAM/Attention Maps).
*   `project_report.md`: This document.
