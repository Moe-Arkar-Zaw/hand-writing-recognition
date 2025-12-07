Hand Writing Project Techniques Summary
1. Dataset Preprocessing

1.1 Vertical Patch Splitting (Image Tripling)
Every training image is divided into 3 vertical sections.
This increases diversity and isolates different parts of the handwriting.

1.2 Image Duplication by Augmentation
For each patch:
 original patch + 1 augmented version = 2 copies
 → With 3 patches → 6 images per original
This directly increases class samples and improves robustness.


2. Image Augmentation (Strong Pre-Segmentation Augmentations)

A powerful augmentation model is applied before segmentation:
RandomRotation(0.20)
RandomTranslation(0.10, 0.10)
RandomZoom(0.15, 0.15)
RandomContrast(0.20)
GaussianNoise(0.02)

Purpose:
Simulate different handwriting variations.
Improve generalization.
Reduce overfitting.


3. Segmentation Pipeline

3.1 Line Segmentation
Uses binary inversion + Otsu threshold + morphological closing.
Horizontal projection profile used to identify text lines.

3.2 Word Segmentation
Additional thresholding + dilation.
Finds contours representing word blocks.

3.3 Character Segmentation
Column projection profile used.
Finds character boundaries using low-ink columns.
Minimum width enforced → ensures single characters extracted.


4. Character Normalization
Before feeding into the CNN:
Convert to grayscale (if needed)
Resize to target_h × target_w (64×64)
Maintain aspect ratio
Center-pad with white pixels
Normalize pixels to [0, 1]
Ensures uniform input shape for the CNN.


5. CNN Model Architecture
A strong custom CNN:
Convolutional Backbone
5 convolutional blocks:
 Filters: 32 → 64 → 128 → 256 → 256

Each block:
Conv2D
BatchNormalization
MaxPool2D
Dropout(0.25)

Character-Level Augmentation (During Training)
Small rotation + translation added to input layer
 → improves robustness at the character-level.

Fully Connected Layers
GlobalAveragePooling2D
Dense(1024, relu) + BatchNorm + Dropout(0.5)
Output softmax for class probability


6. Training Strategy

6.1 Class Weighting
Compensates class imbalance:
 weight = total_samples / (num_classes × class_count)

6.2 Learning Rate Management
ReduceLROnPlateau
Adam(lr=1e-3)

6.3 Regularization Techniques
Batch normalization
Dropout in every conv block
Strong augmentation

6.4 Callbacks
Save best model (val_accuracy)
Early stopping
LR reduction


7. Inference (run.py)

7.1 Apply the Same Segmentation Pipeline
Lines → words → characters

7.2 Predict Characters
Each character passed individually into CNN
Model outputs class probabilities

7.3 Majority Voting
For each test image:

Collect predictions from all characters
Select most frequent predicted class
 → improves robustness if some characters are misread

.4 Output CSV
Stores:
filename
true class
predicted class
Computes final test accuracy

