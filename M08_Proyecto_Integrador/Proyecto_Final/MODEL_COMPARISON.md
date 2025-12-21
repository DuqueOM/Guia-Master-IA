# Model Comparison Report - Fashion-MNIST Analyst

## Executive Summary

This report compares classical linear models vs neural models on **Fashion-MNIST** (28×28 grayscale, 10 classes). The core takeaway is that **linear models struggle on complex image distributions** because real-world visual classes are not separable by a single hyperplane in raw pixel space.

## Dataset

- Dataset: **Fashion-MNIST** (same shape as MNIST, harder classification)
- Input: 28×28 grayscale images
- Flattened representation: 784 features (when using linear/MLP baselines)
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

A (multiclass) logistic regression is a **linear classifier** in pixel space:

- For class `k`: `score_k(x) = w_k^T x + b_k`
- Prediction: `argmax_k score_k(x)`

That means the decision boundaries are unions of hyperplanes. On images, this is a strong limitation.

### 1) Images have nuisance variability that is nonlinear in pixel space
In Fashion-MNIST, the same semantic class can vary by:

- translation/shift
- thickness
- small rotations
- shape deformation
- texture + local contrast

In raw pixels, those variations do not correspond to a simple linear movement that keeps class separation intact.

### 2) The “useful features” are local and compositional
Clothing classes are distinguished by **parts and local patterns** (edges, corners, sleeves, soles) and how they compose.

- Logistic regression sees **all pixels at once** with one weight per pixel.
- It cannot easily express “if there is a collar-like edge pattern *and* a sleeve pattern, then class = Coat” unless this happens to align linearly.

### 3) Overlap is intrinsic (Bayes error is higher than MNIST)
Fashion-MNIST classes are more confusable (e.g., Shirt vs T-shirt/top; Coat vs Pullover). Even the optimal classifier has non-zero error.

Linear models amplify this limitation because:

- they cannot form piecewise nonlinear boundaries
- they cannot build hierarchical invariances

## Why MLP/CNN improve

### MLP
An MLP with ReLU creates **piecewise-linear** decision regions and learns intermediate representations. Even without convolution, the composition of linear layers + nonlinearity expands the set of functions it can represent.

### CNN
A CNN adds inductive bias that matches image structure:

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

## Ablation Studies (required)

Include at least 2 ablations where you change **one thing at a time** and quantify the effect.

| Ablation | Change | Test Accuracy | Δ vs baseline | Interpretation |
|---|---|---:|---:|---|
| Baseline | (your baseline config) | ___ | 0.00 | reference |
| No normalization | X in [0,255] (or missing standardization) | ___ | ___ | explain what changed and why |
| Random init | no Xavier/He | ___ | ___ | explain what changed and why |

## How to reproduce (minimal)

- Train CNN on Fashion-MNIST:

```bash
python3 scripts/train_cnn_pytorch.py --dataset fashion --out artifacts/cnn_fashion.pt
```

- Predict one 28×28 image:

```bash
python3 scripts/predict.py --ckpt artifacts/cnn_fashion.pt --input path/to/image.png
```
