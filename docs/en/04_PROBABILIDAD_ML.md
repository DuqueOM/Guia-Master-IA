# Module 04 - Essential Probability for Machine Learning

> **Week 8 | Prerequisite for Loss Functions, Softmax/Cross-Entropy, and later GMM intuition**
> **Philosophy:** only the probability you need for Line 1 (Machine Learning)

**Language:** English | [Español →](../04_PROBABILIDAD_ML.md)

---

<a id="m04-0"></a>

## How to use this module (0→100 mode)

**Purpose:** connect probability to what you will actually use later:

- cross-entropy as *negative log-likelihood*
- probabilistic classification (logistic/softmax)
- Gaussians as the atom of generative models (GMM intuition)
- numerical stability (avoid `NaN`)

### Learning outcomes (measurable)

By the end of this module you can:

- Explain conditional probability and Bayes’ theorem with a classification example.
- Use the MLE perspective: “choose parameters that make the data most likely”.
- Derive why minimizing cross-entropy = maximizing log-likelihood (binary + multiclass).
- Implement softmax/log-softmax stably (log-sum-exp).
- Diagnose typical numerical failures: `log(0)`, overflow/underflow, probabilities not summing to 1.

Quick references (Spanish source, stable anchors):

- [Resources](RECURSOS.md)
- [Glossary: Binary Cross-Entropy](GLOSARIO.md#binary-cross-entropy)
- [Glossary: Softmax](GLOSARIO.md#softmax)
- [Glossary: Chain Rule](GLOSARIO.md#chain-rule)

### Integration with v4/v5

- Protocols:
  - [Plan v4](PLAN_V4_ESTRATEGICO.md)
  - [Plan v5](PLAN_V5_ESTRATEGICO.md)
- Error log: `study_tools/DIARIO_ERRORES.md`
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M04` in `rubrica.csv`; includes PB-8)

---

## What you should cover (compact scope)

- Probability basics, conditional probability, independence.
- Bayes theorem as a classifier lens.
- Gaussian (univariate + multivariate) as a modeling primitive.
- Maximum Likelihood Estimation (MLE).
- Softmax + log-sum-exp for stability.
- Cross-entropy (binary + categorical).
- Optional “seed” concept: Markov chains (notation + transition matrices).

---

## Deliverables

### E1: `probability.py`

Implement a minimal probability toolkit with NumPy:

- `gaussian_pdf`
- `multivariate_gaussian_pdf`
- `mle_gaussian`
- stable `softmax`
- `cross_entropy` (binary)
- `categorical_cross_entropy` (multiclass)

### E2: Tests

- Add `tests/test_probability.py`.
- Tests should cover:
  - Gaussian PDF sanity (standard normal at 0 ≈ 0.3989)
  - softmax sums to 1 and preserves ordering
  - MLE recovers parameters with enough data
  - cross-entropy near 0 for near-perfect predictions

---

## Consolidation (common errors + debugging)

Common failure patterns:

- Confusing PDF value with probability (continuous vs discrete).
- `log(0)` in cross-entropy (always use `epsilon` / `clip`).
- overflow in `exp` (use log-sum-exp / log-softmax).
- “magic MLE”: if you cannot explain why the mean appears, redo the Bernoulli worked example.

Debugging checklist:

- If something becomes `nan/inf`, check:
  - `np.log` on zeros
  - `np.exp` on large logits
  - probability normalization
- Record cases and fixes in `study_tools/DIARIO_ERRORES.md`.

---

## Feynman challenge (whiteboard)

Explain in 5 lines or less:

1. Why is maximizing likelihood equivalent to minimizing negative log-likelihood?
2. Why is the Bernoulli MLE just “fraction of heads”?
3. What does `π_{t+1} = π_t P` mean and why is it linear algebra?

---

## Completion checklist

- [ ] I can explain Bayes’ theorem with an example.
- [ ] I can compute a Gaussian PDF by hand.
- [ ] I understand why MLE leads to cross-entropy as the loss.
- [ ] I implemented numerically-stable softmax.
- [ ] I can derive the Bernoulli MLE and explain it.
- [ ] I can explain what a Markov chain is and what a transition matrix represents.
- [ ] All tests for `probability.py` pass.

---

## Navigation

**[← Module 03: Calculus](03_CALCULO_MULTIVARIANTE.md)** | **[Module 05: Supervised Learning →](05_SUPERVISED_LEARNING.md)**
