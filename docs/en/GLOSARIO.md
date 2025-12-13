# 📖 Technical Glossary - ML Specialist v3.3

> A–Z definitions of Machine Learning terms used throughout the guide.

**Language:** English | [Español →](../GLOSARIO.md)

---

## A

### Activation Function
**Definition:** Non-linear function applied to a neuron’s output.
**Examples:** ReLU, Sigmoid, Tanh, Softmax.
**Why:** Without activations, a network is just a linear model.

### Adam
**Definition:** Adaptive Moment Estimation - optimizer combining Momentum and RMSprop.
**Typical params:** lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8
**Use:** Common default optimizer for neural networks.

### Accuracy
**Definition:** Fraction of correct predictions.
**Formula:** (TP + TN) / (TP + TN + FP + FN)
**Limitation:** Misleading for imbalanced datasets.

---

## B

### Backpropagation
**Definition:** Algorithm to compute gradients in neural networks using the Chain Rule.
**Process:** Forward pass → compute loss → backward pass → update weights.
**Math base:** ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

### Batch Size
**Definition:** Number of samples processed before updating weights.
**Trade-off:** Large = stable but slower; small = noisy but faster.
**Common:** 32, 64, 128, 256.

### Bias (parameter)
**Definition:** Constant term in z = Wx + b allowing a shift.
**Analogy:** Intercept in y = mx + b.

### Binary Cross-Entropy
**Definition:** Loss function for binary classification.
**Formula:** L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
**Use:** Sigmoid output, probability prediction.

### Broadcasting
**Definition:** Automatic expansion of arrays for elementwise operations.
**Example:** array(3,1) + array(1,4) → array(3,4)
**Rule:** Dimensions must match or one must be 1.

---

## C

### Centroid
**Definition:** Cluster center (mean of its points).
**In K-Means:** updated iteratively until convergence.

### Chain Rule
**Definition:** Rule to differentiate composed functions.
**Formula:** d/dx f(g(x)) = f'(g(x)) · g'(x)
**Importance:** Mathematical basis of backpropagation.

### Classification
**Definition:** Task of predicting a discrete category.
**Binary:** 2 classes (spam/ham).
**Multiclass:** >2 classes (digits 0–9).

### Clustering
**Definition:** Group similar points without labels.
**Algorithms:** K-Means, DBSCAN, Hierarchical.

### Confusion Matrix
**Definition:** Table of predictions vs ground truth.
**Components:** TP, TN, FP, FN.

### Convergence
**Definition:** When an algorithm stops improving meaningfully.
**Criteria:** loss change < tolerance, or gradient ≈ 0.

### Cosine Similarity
**Definition:** Similarity based on the angle between vectors.
**Formula:** cos(θ) = (a·b) / (||a|| ||b||)
**Range:** [-1, 1].

### Cross-Validation
**Definition:** Evaluation by splitting data into K folds.
**K-Fold:** train K times, each time with a different fold as validation.
**Use:** estimate real performance, reduce overfitting risk.

---

## D

### Deep Learning
**Definition:** ML with multi-layer neural networks.
**Advantage:** learns features automatically.
**Requirement:** data + compute.

### Derivative
**Definition:** Instantaneous rate of change.
**Notation:** f'(x), df/dx, ∂f/∂x.

### Dimensionality Reduction
**Definition:** Reduce number of features while preserving information.
**Methods:** PCA, t-SNE, UMAP.
**Use:** visualization, denoising, faster training.

### Dot Product
**Definition:** Sum of elementwise products.
**Formula:** a·b = Σ aᵢbᵢ
**Use:** similarity, projections, neural layers.

---

## E

### Eigenvalue / Eigenvector
**Definition:** For matrix A, Av = λv.
**Interpretation:** stable directions under a linear transform.
**Use in ML:** PCA uses eigenvectors of covariance.

### Epoch
**Definition:** One full pass over the training dataset.
**Typical:** 10–100 (problem-dependent).

### Euclidean Distance
**Definition:** Straight-line distance.
**Formula:** d(a,b) = √Σ(aᵢ - bᵢ)²
**Use:** K-Means, KNN.

---

## F

### F1 Score
**Definition:** Harmonic mean of Precision and Recall.
**Formula:** F1 = 2 · (P · R) / (P + R)

### Feature
**Definition:** Input variable (dataset column).
**Example:** in MNIST, each pixel is a feature (784).

### Forward Pass
**Definition:** propagate input through the network to produce output.
**Computation:** z = Wx + b, a = activation(z), repeat per layer.

---

## G

### Gradient
**Definition:** Vector of partial derivatives.
**Notation:** ∇f = [∂f/∂x₁, ∂f/∂x₂, ...]
**Property:** points to steepest ascent direction.

### Gradient Descent
**Definition:** Optimization algorithm following the negative gradient.
**Update:** θ = θ - α · ∇L(θ)
**Variants:** Batch, Mini-batch, Stochastic (SGD).

---

## H

### Hidden Layer
**Definition:** Layer between input and output.
**Role:** learns intermediate representations.

### Hyperparameter
**Definition:** Parameter set before training (not learned).
**Examples:** learning rate, number of layers, batch size.

---

## I

### Inertia
**Definition:** Sum of squared distances from points to their centroids.
**In K-Means:** objective to minimize.
**Use:** elbow method to pick K.

---

## K

### K-Means
**Definition:** Clustering algorithm partitioning data into K groups.
**Steps:** init centroids → assign → update → repeat.
**Complexity:** O(n · k · i · d)

### K-Means++
**Definition:** Smarter initialization for K-Means.
**Advantage:** better convergence, fewer bad local minima.

---

## L

### L1 Norm (Manhattan)
**Definition:** sum of absolute values.
**Formula:** ||x||₁ = Σ|xᵢ|
**Use:** Lasso regularization, sparsity.

### L2 Norm (Euclidean)
**Definition:** square root of sum of squares.
**Formula:** ||x||₂ = √Σxᵢ²
**Use:** Ridge regularization, normalization.

### Learning Rate
**Definition:** step size in Gradient Descent.
**Symbol:** α or lr.
**Trade-off:** large = fast but unstable; small = stable but slow.

### Linear Regression
**Definition:** predicts a continuous value via a linear combination.
**Formula:** ŷ = Xθ
**Loss:** MSE.

### Logistic Regression
**Definition:** binary classifier using sigmoid.
**Formula:** P(y=1) = σ(Xθ)
**Loss:** Binary Cross-Entropy.

### Loss Function
**Definition:** measures prediction error.
**Examples:** MSE, Cross-Entropy.

---

## M

### Matrix Multiplication
**Definition:** (m×n) @ (n×p) → (m×p)
**Element:** C[i,j] = Σₖ A[i,k] · B[k,j]

### Mini-batch
**Definition:** subset of data used in one SGD update.
**Advantage:** balance between speed and stability.

### MLP (Multilayer Perceptron)
**Definition:** fully-connected neural network with hidden layers.

### MNIST
**Definition:** handwritten digits dataset (28×28).
**Size:** 60k train, 10k test.
**Use:** classic image classification benchmark.

### MSE (Mean Squared Error)
**Definition:** mean of squared errors.
**Formula:** (1/n) Σ(y - ŷ)²

### Momentum
**Definition:** accelerates SGD by accumulating past gradients.
**Formula:** v = β·v + (1-β)·∇L; θ = θ - α·v

---

## N

### Normalization
**Definition:** rescale data to a standard range.
**Min-Max:** (x - min) / (max - min) → [0,1]
**Z-score:** (x - μ) / σ → mean 0, std 1.

### NumPy
**Definition:** Python library for efficient numerical computing.
**Advantage:** vectorized operations (avoid Python loops).
**Core object:** `ndarray`.

---

## O

### One-Hot Encoding
**Definition:** represent a category as a binary vector.
**Example:** class 3 of 5 → [0, 0, 0, 1, 0]

### Overfitting
**Definition:** memorizes training data but fails to generalize.
**Symptom:** low train loss, high test loss.
**Fixes:** more data, regularization, dropout, early stopping.

---

## P

### Partial Derivative
**Definition:** derivative w.r.t. one variable, others constant.
**Notation:** ∂f/∂x

### PCA (Principal Component Analysis)
**Definition:** dimensionality reduction preserving maximum variance.
**Method:** project onto leading eigenvectors / SVD components.
**Output:** principal components ordered by explained variance.

### Precision
**Definition:** among predicted positives, how many are correct?
**Formula:** TP / (TP + FP)

### Projection
**Definition:** map a point onto a subspace (line/plane).
**In PCA:** project data to the principal-component subspace.

---

## R

### Recall
**Definition:** among true positives, how many did we capture?
**Formula:** TP / (TP + FN)

### Regression
**Definition:** predict a continuous value.

### Regularization
**Definition:** prevent overfitting by penalizing complexity.
**L1:** add λ·||θ||₁
**L2:** add λ·||θ||₂²

### ReLU (Rectified Linear Unit)
**Definition:** f(x) = max(0, x)
**Derivative:** 1 if x > 0, else 0.

---

## S

### SGD (Stochastic Gradient Descent)
**Definition:** GD using one sample (or mini-batch) per update.
**Pros:** fast, can escape local minima.
**Cons:** noisy updates.

### Sigmoid
**Definition:** σ(x) = 1 / (1 + e⁻ˣ)
**Range:** (0, 1)
**Use:** binary classification probabilities.

### Silhouette Score
**Definition:** clustering quality metric.
**Range:** [-1, 1] (higher is better).

### Softmax
**Definition:** converts a vector into a probability distribution.
**Formula:** softmax(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ
**Use:** multiclass output layer.

### Supervised Learning
**Definition:** learn from labeled data (X, y).
**Tasks:** classification, regression.

### SVD (Singular Value Decomposition)
**Definition:** factorization A = UΣVᵀ.
**Use:** PCA (more stable), compression, recommender systems.

---

## T

### Tanh
**Definition:** hyperbolic tangent, like sigmoid but centered.
**Range:** (-1, 1)
**Derivative:** 1 - tanh²(x)

### Test Set
**Definition:** held-out data for final evaluation.
**Rule:** never use for training or hyperparameter selection.

### Training Set
**Definition:** data used to fit the model.

### Transpose
**Definition:** swap rows and columns of a matrix.
**Notation:** Aᵀ
**Property:** (AB)ᵀ = BᵀAᵀ

---

## U

### Underfitting
**Definition:** model too simple to capture patterns.
**Symptom:** high train loss, high test loss.

### Unsupervised Learning
**Definition:** learn from unlabeled data.
**Tasks:** clustering, dimensionality reduction, anomaly detection.

---

## V

### Validation Set
**Definition:** data used for hyperparameter tuning.

### Variance (statistics)
**Definition:** dispersion measure.
**Formula:** Var(X) = E[(X - μ)²]

### Variance (ML)
**Definition:** error from sensitivity to training data fluctuations.
**High variance:** overfitting.

### Vectorization
**Definition:** replace loops with array operations.
**Advantage:** 10–100x faster with NumPy.

---

## W

### Weight
**Definition:** learned parameter controlling feature importance.
**In networks:** matrix W in z = Wx + b.

---

## X

### Xavier Initialization
**Definition:** initialize weights with variance 1/n_inputs.
**Form:** W ~ N(0, 1/n_in) or U(-√(1/n_in), √(1/n_in))
**Use:** tanh/sigmoid layers.

### XOR Problem
**Definition:** classic non-linearly separable problem.
**Importance:** shows why hidden layers are needed.
**Solution:** MLP with at least one hidden layer.
