# Aerial GCP Pose Estimation

## Overview

This project implements a computer vision pipeline to automatically detect **Ground Control Point (GCP) markers** in aerial drone imagery.

The system performs **two tasks simultaneously**:

1. **Keypoint Localization** – Predict the pixel coordinates *(x, y)* of the GCP marker center.
2. **Shape Classification** – Classify the marker as one of the following:

   * Cross
   * Square
   * L-Shaped

The model processes aerial images and produces predictions in the required JSON format for evaluation.

---

# Repository Structure

```
gcp-pose-estimation-cv
│
├── dataset
│   ├── gcp_dataset.py
│   └── transforms.py
│
├── models
│   └── gcp_model.py
│
├── train.py
├── inference.py
├── predictions.json
├── requirements.txt
└── README.md
```

---

# Model Architecture

The system uses a **multi-task learning architecture** built on top of a pretrained **ResNet18 backbone**.

### Feature Extraction

```
Input Image (512×512)
        ↓
ResNet18 Backbone
        ↓
Shared Feature Vector
```

### Task Heads

Two task-specific heads are used:

1️⃣ **Keypoint Regression Head**

Predicts the normalized coordinates of the marker center.

```
Linear Layer → (x, y)
```

2️⃣ **Shape Classification Head**

Predicts the marker shape.

```
Linear Layer → 3 classes
```

---

# Training Strategy

### Image Processing

Original aerial images were very large:

```
2048 × 1365
```

To improve training efficiency, all images were **preprocessed and resized to 512×512** before training.

This significantly reduced disk I/O and training time.

---

### Keypoint Normalization

To stabilize training, keypoint coordinates were **normalized to the range [0,1]**.

```
x_norm = x / image_width
y_norm = y / image_height
```

Predictions are converted back to pixel space during inference.

---

### Loss Functions

The model is trained with a **multi-task loss**:

```
Total Loss = 10 × Keypoint MSE + CrossEntropy(shape)
```

* **Keypoint Loss:** Mean Squared Error
* **Shape Loss:** Cross Entropy

Keypoint loss is weighted higher because localization accuracy is the primary evaluation metric.

---

### Optimization

Optimizer used:

```
Adam
learning rate = 3e-4
```

Training configuration:

```
Batch size = 32
Epochs = 20
Image size = 512
```

---

# Handling Dataset Challenges

The dataset contained several real-world inconsistencies:

### Inconsistent Label Names

Some annotations used:

```
"L-Shaped"
```

while others used:

```
"L-Shape"
```

Both were mapped to the same class.

---

### Missing Shape Labels

Some entries lacked the `verified_shape` field.
Fallback logic was implemented to ensure training stability.

---

### Broken Image Paths

Some dataset entries referenced images that failed to load.
The loader skips corrupted samples safely.

---

# Inference

The inference pipeline:

1. Load trained model weights
2. Traverse the test dataset recursively
3. Resize each image to 512×512
4. Predict keypoint and shape
5. Convert normalized coordinates back to pixel coordinates
6. Save predictions in JSON format

Run inference:

```
python inference.py
```

Output:

```
predictions.json
```

---

# Model Weights

Trained model weights are available here:

**[Insert Google Drive Link]**

File:

```
gcp_model.pth
```

---

# Reproducing Results

### Install dependencies

```
pip install -r requirements.txt
```

### Train model

```
python train.py
```

### Generate predictions

```
python inference.py
```

---

# Evaluation Metrics

The model is evaluated using:

### Keypoint Localization

```
Percentage of Correct Keypoints (PCK)
```

This measures whether predicted coordinates fall within a pixel threshold of the true center.

### Shape Classification

```
Macro F1 Score
```

across the three marker shapes.

---

# Future Improvements

Possible improvements include:

* Heatmap-based keypoint detection
* HRNet or EfficientNet backbones
* Multi-scale training
* Stronger data augmentation
* Keypoint heatmap supervision

---

# Author

Askari Abidi
B.Tech CSE (AI & ML)
Christ (Deemed to be University)
