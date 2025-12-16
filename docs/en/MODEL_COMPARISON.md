# Model Comparison Report - Fashion-MNIST Analyst

## Executive Summary

This report compares a linear baseline vs neural models on **Fashion-MNIST** (28×28 grayscale, 10 classes). The key conclusion is that **linear models struggle on complex image distributions** because real-world visual classes are not separable by a single hyperplane in raw pixel space.

## Dataset

- Dataset: **Fashion-MNIST** (same shape as MNIST, harder classification)
- Input: 28×28 grayscale images
- Flattened representation: 784 features (for linear/MLP baselines)
- Classes (0–9): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Models

- Logistic Regression (baseline)
- MLP (manual backprop, NumPy)
- CNN (PyTorch training, reference)

## Experimental Setup (minimum reproducible)

- Train/test split: official dataset split (60k/10k)
- Normalization: scale to `[0, 1]`
- Metric: accuracy (plus per-class analysis)

## Results (fill with your actual run)

| Model | Dataset | Test Accuracy | Notes |
|---|---|---:|---|
| Logistic Regression (OvA) | Fashion-MNIST | ___ | linear baseline |
| MLP (784→128→64→10) | Fashion-MNIST | ___ | nonlinear, learned features |
| CNN (SimpleCNN_v1) | Fashion-MNIST | ___ | local receptive fields + pooling |

## Why Linear Models Fail on Complex Images

Multiclass logistic regression is a **linear classifier** in pixel space:

- For class `k`: `score_k(x) = w_k^T x + b_k`
- Prediction: `argmax_k score_k(x)`

So decision boundaries are unions of hyperplanes. On images, that is a strong limitation.

### 1) Nuisance variability is nonlinear in pixel space
In Fashion-MNIST, the same semantic class can vary by:

- translation/shift
- thickness
- small rotations
- shape deformation
- texture + local contrast

In raw pixels, those variations do not correspond to a simple linear change that preserves class separation.

### 2) Useful features are local and compositional
Clothing classes are distinguished by **parts and local patterns** (edges, corners, sleeves, soles) and their composition.

- Logistic regression assigns one weight per pixel and has no notion of locality.
- It cannot easily implement rules like “collar pattern AND sleeve pattern ⇒ Coat” unless it is linearly separable.

### 3) Intrinsic overlap (higher Bayes error than MNIST)
Fashion-MNIST is more ambiguous than MNIST (e.g., Shirt vs T-shirt/top; Coat vs Pullover). Even the best classifier will make mistakes.

Linear models amplify this limitation because:

- they cannot form piecewise nonlinear boundaries
- they cannot build hierarchical invariances

## Why MLP/CNN improve

### MLP
An MLP with ReLU creates **piecewise-linear** decision regions and learns intermediate representations. Even without convolution, composing linear maps with nonlinearities dramatically increases expressivity.

### CNN
A CNN adds inductive bias aligned with image structure:

- local receptive fields
- parameter sharing
- translation tolerance (pooling)

This typically yields a larger improvement on Fashion-MNIST than on MNIST.

## Error Analysis (required)

Include:

- Confusion matrix (10×10)
- Top confusions and a short explanation (e.g., Shirt vs T-shirt/top)
- A grid of misclassified images with predicted label + confidence

## Bias–Variance Diagnosis

Report at least one learning-curve experiment:

- train on {1k, 5k, 10k, full} samples
- plot train vs test accuracy for Logistic vs MLP
- diagnose which side (bias/variance) is limiting each model

## How to reproduce (minimal)

- Train CNN on Fashion-MNIST:

```bash
python3 scripts/train_cnn_pytorch.py --dataset fashion --out artifacts/cnn_fashion.pt
```

- Predict one 28×28 image:

```bash
python3 scripts/predict.py --ckpt artifacts/cnn_fashion.pt --input path/to/image.png
```
