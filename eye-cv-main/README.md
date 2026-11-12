# EyeCV — Retinal Disease Classification

**Utkarsh Mishra** — [@devutkarshh](https://github.com/devutkarshh)

An end-to-end deep-learning system that automatically classifies retinal (fundus)
photographs into four eye conditions: **cataract**, **diabetic retinopathy**,
**glaucoma**, and **normal**. The project implements and benchmarks three
state-of-the-art architectures — ResNet50, EfficientNetB0, and a Vision
Transformer (ViT-B16) — and ships a dedicated explainability module (Grad-CAM and
Transformer attention maps) so every prediction can be visually inspected.

---

## Overview

1. Load and enhance retinal images with a medical-image preprocessing pipeline.
2. Fine-tune three ImageNet-pretrained architectures on a 4-class classification task.
3. Evaluate each model with precision, recall, F1, and confusion matrices.
4. Explain predictions visually with Grad-CAM (CNNs) and attention maps (ViT).

## Dataset

| Property | Value |
|---|---|
| Classes | `cataract`, `diabetic_retinopathy`, `glaucoma`, `normal` |
| Total images | **4,217** RGB fundus photographs |
| Split | 80% train / 20% validation (`seed=42`) |

The dataset is relatively balanced (~1,000 images per class). See
`DATASET_DESCRIPTION.md` for exact counts and image properties.

## Preprocessing

All images are resized to **224×224** and run through a medical enhancement
pipeline designed to make subtle retinal features more visible to the model:

1. Convert RGB → CIE-LAB color space.
2. Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to the
   **L-channel** (`clipLimit=2.0`, `tileGridSize=(8,8)`) — this enhances
   vessels, exudates, and microaneurysms while suppressing noise.
3. Convert back to RGB and normalize to `[0, 1]`.

During training only, random **horizontal/vertical flip**, **rotation**,
**zoom**, and **contrast** augmentation are applied.

## Models

Transfer learning from ImageNet is used for all three architectures. Each model
adds a `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(num_classes, softmax)`
classification head.

| Architecture | Family | Backed by |
|---|---|---|
| ResNet50 | CNN with residual connections | `tf.keras.applications.ResNet50` |
| EfficientNetB0 | Efficient compound-scaled CNN | `tf.keras.applications.EfficientNetB0` |
| ViT-B16 | Vision Transformer | Hugging Face `google/vit-base-patch16-224` |

> The ViT pipeline reshapes inputs to channels-first `(C,H,W)` and wraps the
> Hugging Face model in a custom `HFViTWrapper` Keras layer so it can be trained
> inside a standard Keras `Model`.

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam (`learning_rate=1e-4`) |
| Loss | Categorical crossentropy |
| Batch size | 32 |
| Epochs | 30 |
| Sensitivity metric | Recall (compiled as `"sensitivity"`) |
| Callbacks | `ModelCheckpoint` (best `val_accuracy`), `EarlyStopping` (patience=10), `ReduceLROnPlateau` (factor=0.5, patience=3) |
| Data pipeline | `tf.data` with `AUTOTUNE` parallelism and `prefetch` |

## Results

Validation accuracy and per-class recall (full `classification_report` +
`confusion_matrix` are printed and plotted by each training script):

| Model | Validation Accuracy | Best-class Recall | Worst-class Recall |
|---|---|---|---|
| **ResNet50** | **95%** | Normal (96%) | Glaucoma (93%) |
| **ViT-B16** | **94%** | Diabetic Retinopathy (95%) | Normal (87%) |
| **EfficientNetB0** | **87%** | Cataract (93%) | Diabetic Retinopathy (81%) |

ResNet50 and ViT-B16 are the top performers; EfficientNetB0 is the lightest but
lags on the finer-grained DR/Glaucoma distinctions. See `project_report.md` for
the full analysis.

## Explainability

`colab_explainability.py` loads a saved model and visualizes what it "looks at":

- **ResNet50** → **Grad-CAM** over the `conv5_block3_out` layer using
  `tf.GradientTape`, overlaid on the original image.
- **ViT-B16** → **class attention map** averaged across heads from the last
  Transformer layer, upscaled and overlaid on the input.

```bash
python colab_explainability.py --model resnet   # Grad-CAM
python colab_explainability.py --model vit      # Attention map
```

## Project Structure

```
EyeCV/
├── colab_resnet_full.py          # ResNet50: preprocess -> train -> evaluate
├── colab_efficientnet_full.py    # EfficientNetB0: preprocess -> train -> evaluate
├── colab_vit_full.py             # ViT-B16 (Hugging Face Transformers) pipeline
├── colab_explainability.py       # Grad-CAM + ViT attention visualizations
├── requirements.txt              # Python dependencies
├── DATASET_DESCRIPTION.md        # Dataset documentation (4 classes, 4,217 images)
├── project_report.md             # Full written project report and analysis
└── explainability_output.txt     # Sample explainability run log
```

## Requirements

```
transformers
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
```

Install:

```bash
pip install -r requirements.txt
```

## How to Run

> Tip: these pipelines are designed for Google Colab (GPU). Place the dataset
> folder (containing the 4 class subdirectories) at `dataset/` (or
> `Data/dataset/`).

Train a model:

```bash
python colab_resnet_full.py        # saves resnet50_final.keras
python colab_efficientnet_full.py  # saves efficientnet_final.keras
python colab_vit_full.py           # saves vit_final.weights.h5
```

Explain a trained model:

```bash
python colab_explainability.py --model resnet   # or --model vit
```

## Reproducibility

- Fixed splits and seeds (`seed=42` for training, `seed=123` for the explainability
  validation set).
- Deterministic data augmentation is applied only to the training split.
- Best-checkpoint weights are restored before evaluation.

## Author

**Utkarsh Mishra** — [github.com/devutkarshh](https://github.com/devutkarshh)
#
