# Decision Log

This section documents key engineering decisions made during the project.

---

## 1. Choice of Model Architecture

A **ResNet18 backbone** was selected because:

* Strong performance on image tasks
* Lightweight and fast to train
* Pretrained weights available
* Easy to extend with multi-task heads

More complex architectures (HRNet, EfficientNet) were considered but would increase complexity without clear benefits for this dataset size.

---

## 2. Multi-Task Learning

Instead of training separate models, a **shared backbone with two heads** was implemented.

Benefits:

* Shared visual features improve learning
* Reduced computational cost
* Simpler inference pipeline

---

## 3. Image Preprocessing Strategy

The original images were very large:

```
2048 × 1365
```

Loading and resizing these images during every training iteration caused severe I/O bottlenecks.

To resolve this, a preprocessing step was implemented to resize images to **512×512 before training**.

This reduced training time by approximately **10–20×**.

---

## 4. Keypoint Normalization

Initial training used raw pixel coordinates which caused extremely large regression losses.

Coordinates were normalized to **[0,1]** to stabilize optimization and improve convergence.

---

## 5. Robust Dataset Loader

Because the dataset reflects real production conditions, the loader includes safeguards:

* Handles missing shape labels
* Handles inconsistent label names
* Skips corrupted images

This ensures training does not crash due to imperfect annotations.

---

## 6. Loss Balancing

Localization accuracy is more important than classification for this task.

Therefore the loss was balanced as:

```
Total Loss = 10 × Keypoint Loss + Shape Loss
```

This encourages the model to prioritize accurate center prediction.

---

## 7. Pretrained Weights

The ResNet18 backbone uses **ImageNet pretrained weights**, which significantly improves performance given the relatively small dataset (1000 images).
