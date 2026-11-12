# ==========================================
# Vision Transformer (ViT) - Full Pipeline for Google Colab
# ==========================================
# Instructions:
# 1. Install dependencies: !pip install transformers
# 2. Upload your 'dataset.zip' to Colab.
# 3. Run: !unzip -q dataset.zip -d dataset
# 4. Run this cell.

import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TFViTModel

# -------------------------------------------------------------------
# 0. Data Augmentation
# -------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# -------------------------------------------------------------------
# 1. Data Loading & Preprocessing (CLAHE)
# -------------------------------------------------------------------

def medical_preprocessing(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the Green channel
    of the image to enhance retinal features (vessels, lesions).
    """
    # Convert tensor to numpy for OpenCV CLAHE
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Ensure image is uint8 for OpenCV
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Check if image is grayscale (1 channel) or RGB (3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply CLAHE to the Green Channel (Retinal features are clearest here)
    # We convert to LAB to apply CLAHE to L channel as per standard practice for contrast
    # (or we could extract Green channel from RGB, apply CLAHE, and merge back, but LAB is better for contrast)
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except cv2.error as e:
        print(f"OpenCV Error in medical_preprocessing: {e}")
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        # Return original image if CLAHE fails to avoid crashing
        return image / 255.0

    return final_img / 255.0  # Normalize to [0,1]

def preprocess_wrapper(image, label):
    """
    Wraps the numpy function for use in tf.data pipeline.
    """
    # tf.py_function is needed because we use cv2 (numpy)
    [processed_img] = tf.py_function(medical_preprocessing, [image], [tf.float32])

    # Explicitly set shape because py_function loses it
    processed_img.set_shape([224, 224, 3])
    return processed_img, label

def get_datasets(data_dir, batch_size=32, image_size=(224, 224), seed=42):
    """
    Loads dataset from directory and splits into train, validation, and test.
    """

    # Load Training Data (80%)
    # Set batch_size=None to load individual images for preprocessing
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=None,
        label_mode='categorical'
    )

    # Load Validation Data (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=None,
        label_mode='categorical'
    )

    # Capture class names before transformation
    class_names = train_ds.class_names

    # Apply Preprocessing to individual images
    train_ds = train_ds.map(preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply Data Augmentation to Training Data ONLY
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

# -------------------------------------------------------------------
# 2. Model Definition (ViT-B16 with Transformers)
# -------------------------------------------------------------------

# Custom Layer to wrap Hugging Face model
class HFViTWrapper(tf.keras.layers.Layer):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs):
        # HF model returns a TFBaseModelOutputWithPooling, we want the pooler_output
        return self.base_model(pixel_values=inputs).pooler_output

    def get_config(self):
        config = super().get_config()
        return config

def build_vit_b16(num_classes, input_shape=(224, 224, 3)):
    """
    Builds a Vision Transformer (ViT-B16) model using Hugging Face Transformers.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # 1. Preprocessing for ViT (google/vit-base-patch16-224)
    # Expects values in [-1, 1] (mean 0.5, std 0.5)
    # Input is [0, 1]
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    
    # 2. Transpose to Channels First (NCHW)
    # HF ViT expects (batch, channels, height, width)
    x = tf.keras.layers.Permute((3, 1, 2))(x)
    
    # 3. Load Pre-trained Model
    # We use 'google/vit-base-patch16-224'
    # Setting use_safetensors=False to avoid "safe_open object is not iterable" error in some TF/Transformers versions
    base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224', use_safetensors=False)
    # Start Unfrozen (Trainable)
    base_model.trainable = True
    
    # Use custom layer to wrap HF model (avoids Lambda layer variable tracking warnings)
    x = HFViTWrapper(base_model)(x)
    
    # 4. Classification Head
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions, name="ViT-B16")
    return model

# -------------------------------------------------------------------
# 3. Training & Evaluation
# -------------------------------------------------------------------

def run_pipeline(data_dir, epochs=20):
    # 1. Load Data
    train_ds, val_ds, class_names = get_datasets(data_dir)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # 2. Build Model
    model = build_vit_b16(num_classes)
    
    # Use a standard learning rate for Adam
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=["accuracy", tf.keras.metrics.Recall(name="sensitivity")]
    )
    
    # 3. Train
    checkpoint_path = "vit_best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]
    
    print("Starting training (Unfrozen with Augmentation)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    # Save final model weights
    model.save_weights("vit_final.weights.h5")
    print("Final model weights saved to vit_final.weights.h5")
    
    # 4. Evaluate
    print("\nRunning Evaluation...")
    
    # Clear session to reset layer naming (critical for loading weights by name)
    tf.keras.backend.clear_session()
    
    # Rebuild model and load weights
    # We don't strictly need custom_objects for load_weights, but it's good practice if we were loading a full model
    model = build_vit_b16(num_classes)
    model.load_weights(checkpoint_path)
    
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ViT-B16')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "dataset" 
    
    # Check for Google Drive path
    if os.path.exists('/content/drive/MyDrive/dataset'):
        DATA_DIR = '/content/drive/MyDrive/dataset'
    
    if os.path.exists(DATA_DIR):
        if os.path.exists(os.path.join(DATA_DIR, 'dataset')):
             DATA_DIR = os.path.join(DATA_DIR, 'dataset')
             
        run_pipeline(DATA_DIR, epochs=30)
    else:
        print(f"Error: {DATA_DIR} not found. Please upload and unzip your dataset or mount Google Drive.")
