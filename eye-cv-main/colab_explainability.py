# ==========================================
# Explainability Pipeline (Grad-CAM & Attention Maps)
# ==========================================
# Instructions:
# 1. Ensure 'resnet50_final.keras' and 'vit_final.weights.h5' are present.
# 2. Run this script to visualize what the models are looking at.

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from transformers import TFViTModel

# Force CPU usage if desired (optional, TF handles this automatically usually)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# -------------------------------------------------------------------
# 1. Preprocessing (Same as Training)
# -------------------------------------------------------------------
def medical_preprocessing(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except:
        return image / 255.0

    return final_img / 255.0

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img = medical_preprocessing(img)
    return np.expand_dims(processed_img, axis=0), img

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
def setup_logging(log_file="explainability_output.txt"):
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file, mode='w')
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    # Check if handlers already exist to avoid duplicates in interactive environments
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    # Suppress verbose logs from libraries
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('h5py').setLevel(logging.ERROR)
    
    return logger

logger = setup_logging()

# -------------------------------------------------------------------
# ViT Wrapper
# -------------------------------------------------------------------
class HFViTWrapper(tf.keras.layers.Layer):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
    def call(self, inputs):
        return self.base_model(pixel_values=inputs).pooler_output
    def get_config(self):
        return super().get_config()

# -------------------------------------------------------------------
# 4. Main Pipeline: Dataset -> Evaluate -> Sample -> Explain
# -------------------------------------------------------------------
def get_validation_dataset(data_dir):
    logger.info(f"Loading dataset from {data_dir}...")
    try:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(224, 224),
            batch_size=32,
            label_mode='categorical'
        )
        return val_ds
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def explain_batch(model, model_type, images, labels, class_names):
    num_to_explain = min(3, len(images))
    
    for i in range(num_to_explain):
        img_tensor = images[i]
        label_tensor = labels[i]
        
        true_index = np.argmax(label_tensor)
        true_label = class_names[true_index]
        
        img_batch = tf.expand_dims(img_tensor, axis=0)
        
        preds = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx]
        pred_label = class_names[pred_idx]
        
        logger.info(f"Image {i+1}: True={true_label}, Pred={pred_label} ({confidence:.2%})")
        
        original_img = img_tensor.numpy().astype("uint8")
        
        if model_type == "resnet":
            try:
                base_model = model.layers[1]
                last_conv_layer_name = "conv5_block3_out"
                
                grad_model = tf.keras.Model(
                    [base_model.input], 
                    [base_model.get_layer(last_conv_layer_name).output]
                )
                
                with tf.GradientTape() as tape:
                    conv_outputs = grad_model(img_batch)
                    tape.watch(conv_outputs)
                    x = model.layers[2](conv_outputs)
                    x = model.layers[3](x)
                    final_output = model.layers[4](x)
                    score = final_output[:, pred_idx]
                
                grads = tape.gradient(score, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                heatmap = heatmap.numpy()
                
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                superimposed_img = heatmap * 0.4 + original_img
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(original_img)
                plt.title(f"True: {true_label}")
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(superimposed_img)
                plt.title(f"Grad-CAM\nPred: {pred_label}")
                plt.axis('off')
                plt.show()
                logger.info("Displayed Grad-CAM visualization.")
            except Exception as e:
                logger.error(f"Grad-CAM failed: {e}")
            
        elif model_type == "vit":
            try:
                wrapper_layer = None
                for layer in model.layers:
                    if "hf_vi_t_wrapper" in layer.name or isinstance(layer, HFViTWrapper):
                        wrapper_layer = layer
                        break
                
                if wrapper_layer:
                    base_model_hf = wrapper_layer.base_model
                    x_in = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(img_batch)
                    x_in = tf.keras.layers.Permute((3, 1, 2))(x_in)
                    
                    outputs = base_model_hf(pixel_values=x_in, output_attentions=True)
                    attentions = outputs.attentions
                    last_layer_att = attentions[-1]
                    att_mean = tf.reduce_mean(last_layer_att, axis=1)
                    cls_att = att_mean[0, 0, 1:]
                    grid_size = int(np.sqrt(196))
                    att_map = tf.reshape(cls_att, (grid_size, grid_size)).numpy()
                    att_map = cv2.resize(att_map, (224, 224))
                    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
                    
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_img)
                    plt.title(f"True: {true_label}")
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(original_img)
                    plt.imshow(att_map, cmap='jet', alpha=0.5)
                    plt.title(f"Attention\nPred: {pred_label}")
                    plt.axis('off')
                    plt.show()
                    logger.info("Displayed Attention Map visualization.")
                else:
                    logger.warning("Could not find ViT wrapper layer for attention extraction.")
            except Exception as e:
                logger.error(f"ViT Attention failed: {e}")

def run_pipeline(model_type="resnet"):
    data_dir = "Data/dataset"
    if not os.path.exists(data_dir):
        data_dir = "dataset"
    
    if not os.path.exists(data_dir):
        logger.error("Dataset directory not found.")
        return

    val_ds = get_validation_dataset(data_dir)
    if not val_ds:
        return
    
    class_names = val_ds.class_names
    logger.info(f"Classes: {class_names}")

    if model_type == "resnet":
        model_path = os.path.join("models", "resnet50_final.keras")
        if not os.path.exists(model_path): model_path = "resnet50_final.keras"
        if not os.path.exists(model_path):
            logger.error("ResNet model not found.")
            return
        logger.info(f"Loading ResNet from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
    elif model_type == "vit":
        weights_path = os.path.join("models", "vit_final.weights.h5")
        if not os.path.exists(weights_path): weights_path = "vit_final.weights.h5"
        if not os.path.exists(weights_path):
            logger.error("ViT weights not found.")
            return
            
        logger.info(f"Loading ViT from {weights_path}...")
        base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224', output_attentions=True, use_safetensors=False)
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
        x = tf.keras.layers.Permute((3, 1, 2))(x)
        x_wrapper = HFViTWrapper(base_model)(x)
        x_cls = tf.keras.layers.Dense(256, activation='relu')(x_wrapper)
        x_cls = tf.keras.layers.Dropout(0.2)(x_cls)
        outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x_cls)
        model = tf.keras.Model(inputs, outputs)
        model.load_weights(weights_path)

    logger.info("Evaluating on Validation Set...")
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = model.evaluate(val_ds, verbose=1)
    logger.info(f"Validation Accuracy: {acc:.2%}")

    logger.info("Sampling a batch for explanation...")
    for images, labels in val_ds.take(1):
        explain_batch(model, model_type, images, labels, class_names)
    
    logger.info("Pipeline Completed Successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "vit"])
    args = parser.parse_args()
    
    run_pipeline(args.model)
