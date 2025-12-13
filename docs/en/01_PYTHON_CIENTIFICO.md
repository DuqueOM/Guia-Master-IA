# Module 01 - Scientific Python + Pandas

> **Goal:** Use Pandas for real-world data + NumPy for math-ready arrays.
> **Phase:** 1 - Foundations | **Weeks 1–2**
> **Prerequisites:** Basic Python (variables, functions, lists, loops)

**Language:** English | [Español →](../01_PYTHON_CIENTIFICO.md)

---

<a id="m01-0"></a>

## How to use this module (0→100 mode)

**Purpose:** move from “I know basic Python” to **handling real datasets** and producing clean `np.ndarray` inputs (`X`, `y`) with correct shapes and dtypes.

### Learning outcomes (measurable)

By the end of this module you can:

- Load, explore, and clean real datasets with Pandas.
- Convert datasets to `np.ndarray` with correct ML shapes.
- Explain vectorization and why NumPy avoids Python loops.
- Predict and debug common shape issues (`(n,)` vs `(n, 1)`, broadcasting surprises, views vs copies).

### Prerequisites

- Basic Python control flow and functions.

Quick references (Spanish source, stable anchors):

- [Glossary: NumPy](GLOSARIO.md#numpy)
- [Glossary: Broadcasting](GLOSARIO.md#broadcasting)
- [Glossary: Vectorization](GLOSARIO.md#vectorization)
- [Resources](RECURSOS.md)

### How this integrates with v4/v5

- Daily shape drill: `study_tools/DRILL_DIMENSIONES_NUMPY.md`
- Error log: `study_tools/DIARIO_ERRORES.md`
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M01` in `rubrica.csv`)
- Execution protocols:
  - [Plan v4](PLAN_V4_ESTRATEGICO.md)
  - [Plan v5](PLAN_V5_ESTRATEGICO.md)

---

## Recommended resources (when to use them)

| Priority | Resource | When | Why |
|---|---|---|---|
| Required | Pandas Getting Started | Week 1 | Canonical loading/EDA/cleaning workflows |
| Required | NumPy docs (absolute beginners) | Week 2 | Shapes, `axis`, broadcasting, `dtype` |
| Required | `study_tools/DRILL_DIMENSIONES_NUMPY.md` | Any time shapes hurt | Automate shape intuition |
| Optional | Real Python - NumPy | After broadcasting basics | Practical idioms + patterns |

---

## Module content (2-week plan)

### Week 1: Pandas + basic NumPy

- DataFrame/Series fundamentals.
- Reading CSVs (`read_csv`, `head`, `info`).
- Cleaning basics (`dropna`, `fillna`, dtypes).
- Converting Pandas → NumPy (`to_numpy`, shapes).

### Week 2: Vectorized NumPy (performance + correctness)

- Broadcasting rules (predict before running).
- Matrix product (`@`, `np.dot`, `np.matmul`).
- Reshaping (`reshape`, `flatten`, `transpose`).
- Axis-based reductions (mean/sum per axis).
- Synthetic data with RNG (reproducibility).

---

## Key concepts you must own

- **Shape-first thinking**:
  - Always know the expected `(n_samples, n_features)` and `(n_samples,)`/`(n_samples, 1)` targets.
- **Vectorization**:
  - Replace Python loops with NumPy ops; know why it is faster (C loops + contiguous memory).
- **Copy vs view**:
  - Know when modifying an array mutates the original.

---

## Deliverable (Module 01)

### Script: `benchmark_vectorization.py`

**Goal:** compare pure-Python list operations vs NumPy vectorized operations.

Minimum requirements:

- Dot product.
- Normalization.
- Euclidean distance.
- Matrix sum.

Success criteria:

- Demonstrate **>50x** speedup in at least one realistic setting.
- Add at least 3 `pytest` tests.
- Typecheck and lint checks pass (`mypy`, `ruff`).

---

## Whiteboard challenge (Feynman)

Explain in 5 lines or less:

1. Why is NumPy faster than Python lists?
2. What does `axis=0` vs `axis=1` mean?
3. Why does `.copy()` matter?

---

## Completion checklist (v3.2)

### Knowledge

- [ ] I can create 1D/2D/3D arrays with NumPy.
- [ ] I understand indexing and slicing.
- [ ] I can explain and use broadcasting.
- [ ] I can compute reductions by axis.
- [ ] I can rewrite loops as vectorized operations.
- [ ] I know the difference between `@`, `np.dot`, `np.matmul`.
- [ ] I know the 5 most common NumPy errors and how to fix them.

### Code deliverables

- [ ] `benchmark_vectorization.py` implemented.
- [ ] Speedup (NumPy vs list) is >50x in my tests.
- [ ] `mypy` passes.
- [ ] `ruff` passes.
- [ ] At least 3 `pytest` tests passing.

### Feynman

- [ ] I can explain broadcasting in 5 lines.
- [ ] I can explain `axis=0` vs `axis=1` in 5 lines.
- [ ] I can explain why `.copy()` matters.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| - | [00_INDICE](00_INDICE.md) | [02_ALGEBRA_LINEAL_ML](02_ALGEBRA_LINEAL_ML.md) |
