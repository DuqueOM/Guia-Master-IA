# Module 07 - Deep Learning

> **Goal:** implement an MLP with manual backprop + CNN forward (NumPy) + CNN training with PyTorch.
> **Phase:** 2 - ML Core | **Weeks 17–20**
> **Pathway course:** Introduction to Deep Learning

**Language:** English | [Español →](../07_DEEP_LEARNING.md)

---

<a id="m07-0"></a>

## How to use this module (0→100 mode)

**Purpose:** build and debug a neural network from scratch:

- forward pass
- backpropagation
- optimization (SGD / Momentum / Adam)
- sanity checks (overfit test)

### Learning outcomes (measurable)

By the end of this module you can:

- Implement an MLP that solves XOR.
- Explain backprop as Chain Rule applied to a computation graph.
- Debug training using an overfit test (if it cannot memorize, there is a bug).
- Implement the forward pass of a simple CNN (convolution + pooling) in NumPy to master shapes.
- Train an equivalent CNN using PyTorch (`torch.nn`) **without** implementing a manual CNN backward pass.

Quick references:

- [Module 03: Calculus (Chain Rule)](03_CALCULO_MULTIVARIANTE.md)
- [Glossary](GLOSARIO.md)
- [Resources](RECURSOS.md)
- [Plan v4](PLAN_V4_ESTRATEGICO.md)
- [Plan v5](PLAN_V5_ESTRATEGICO.md)
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M07` in `rubrica.csv`; closes Week 20)

---

## Module structure (Weeks 17–20)

| Week | Focus | Output |
|---|---|---|
| 17 | Perceptron + activations + MLP forward pass | `activations.py` + forward utilities |
| 18 | Backpropagation (manual) | `backward()` + gradients |
| 19 | **CNNs: theory + forward (NumPy)** | conv/pooling forward + shape quiz |
| 20 | **PyTorch for CNN training** | `../scripts/train_cnn_pytorch.py` |

---

## What matters most (high-signal core)

### MLP correctness

- Your MLP must pass basic sanity checks:
  - gradients have correct shapes
  - training decreases loss
  - it can overfit a tiny dataset

### CNN split approach (v3.3)

- **NumPy:** forward pass only (shape mastery, not a full framework).
- **PyTorch:** full CNN training loop (realistic workflow).

This preserves the “from scratch” learning goal while avoiding a large, error-prone manual CNN backward.

---

## Deliverables

- `neural_network.py`:
  - MLP forward + backward
  - SGD / Momentum / Adam
  - XOR training demo

- `overfit_test.py` (required):
  - memorization test on a tiny dataset (XOR or small batch)

- CNN practical training:
  - run: `scripts/train_cnn_pytorch.py`

Minimum acceptance criteria:

- XOR is solved reliably.
- Overfit test reaches near-zero loss.
- You can explain each piece in plain language.

---

## Completion checklist (v3.3)

### Knowledge

- [ ] Understand biological neuron → artificial neuron analogy.
- [ ] Implemented sigmoid, ReLU, tanh, softmax and derivatives.
- [ ] Understand why XOR is not linearly separable.
- [ ] Implemented MLP forward pass.
- [ ] Understand Chain Rule applied to backprop.
- [ ] Implemented backward pass (gradients).
- [ ] Implemented SGD, SGD+Momentum, Adam.
- [ ] Network solves XOR.

### CNNs (theory)

- [ ] Understand convolution, stride, padding, pooling.
- [ ] Can compute CNN output dimensions.
- [ ] Know LeNet-5 at a concept level.

### CNNs (practice)

- [ ] Implemented NumPy forward pass (conv + pooling) for a LeNet-like architecture.
- [ ] Trained an equivalent CNN with PyTorch using `scripts/train_cnn_pytorch.py`.

### Code deliverables

- [ ] `neural_network.py` tests passing.
- [ ] `mypy` passes.
- [ ] `pytest` passes.

### Overfit test (required v3.3)

- [ ] `overfit_test.py` implemented.
- [ ] Network overfits XOR (loss < 0.01).
- [ ] If it failed, I debugged using gradient checks.

### Analytical derivation (required)

- [ ] Derived backprop equations by hand.
- [ ] Documented a computation graph.

### Feynman

- [ ] Explain backprop in 5 lines.
- [ ] Explain ReLU vs sigmoid in 5 lines.
- [ ] Explain convolution in 5 lines.
- [ ] Explain pooling in 5 lines.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [08_PROYECTO_MNIST](08_PROYECTO_MNIST.md) |
