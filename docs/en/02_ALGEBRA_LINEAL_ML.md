# Module 02 - Linear Algebra for Machine Learning

> **Goal:** master vectors, matrices, norms, and eigen concepts used throughout ML.
> **Phase:** 1 - Math Foundations | **Weeks 3–5**
> **Prerequisites:** Module 01 (NumPy fundamentals)

**Language:** English | [Español →](../02_ALGEBRA_LINEAL_ML.md)

---

<a id="m02-0"></a>

## How to use this module (0→100 mode)

**Purpose:** learn the “grammar” of ML math so you can read/write expressions like:

- `ŷ = Xθ` (supervised models)
- projections and bases (PCA)
- decompositions (SVD)

### Learning outcomes (measurable)

By the end of this module you can:

- Use dot product and cosine similarity as a geometric similarity signal.
- Implement norms/distances (L1/L2/L∞) and explain their role in regularization.
- Reason about shapes to avoid silent bugs.
- Explain eigenvectors/eigenvalues as “stable directions” and connect them to PCA.
- Explain SVD and why it is the numerically stable way to implement PCA.

Quick references (Spanish source, stable anchors):

- [Glossary: Dot Product](GLOSARIO.md#dot-product)
- [Glossary: Matrix Multiplication](GLOSARIO.md#matrix-multiplication)
- [Glossary: L1 Norm](GLOSARIO.md#l1-norm-manhattan)
- [Glossary: L2 Norm](GLOSARIO.md#l2-norm-euclidean)
- [Glossary: SVD](GLOSARIO.md#svd-singular-value-decomposition)
- [Resources](RECURSOS.md)

### Integration with v4/v5

- Daily shape drill: `study_tools/DRILL_DIMENSIONES_NUMPY.md`
- Weekly exam drills: `study_tools/SIMULACRO_EXAMEN_TEORICO.md`
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M02` in `rubrica.csv`)
- Protocols:
  - [Plan v4](PLAN_V4_ESTRATEGICO.md)
  - [Plan v5](PLAN_V5_ESTRATEGICO.md)

---

## Recommended resources (when to use them)

| Priority | Resource | When | Why |
|---|---|---|---|
| Required | `study_tools/DRILL_DIMENSIONES_NUMPY.md` | Any time shapes go wrong | Prevent silent broadcasting bugs |
| Required | 3Blue1Brown: Linear Algebra | Weeks 3–4 | Build geometric intuition |
| Optional | Mathematics for ML (Linear Algebra) | Week 5 | Formal exercises |

---

## Module content (3-week plan)

- **Week 3:** vectors, dot product, projections, cosine similarity.
- **Week 4:** norms and distances, geometry + regularization intuition.
- **Week 5:** matrices, determinants, eigen basics, and SVD.

What matters most:

- You can predict shapes.
- You can explain the geometry.
- You can connect eigen/SVD to PCA (Module 06).

---

## Deliverable (Module 02)

### Library: `linear_algebra.py`

Implement a small, testable library that covers:

- Dot product, cosine similarity, projections.
- Norms (L1/L2/L∞), normalization.
- Distances (euclidean, manhattan, cosine distance).
- Pairwise distances.
- PCA building blocks (eigen/SVD-level primitives).

Practical acceptance criteria:

- Functions return correct shapes and types.
- Unit tests exist for the core primitives.
- You can use these utilities later in PCA/K-Means.

---

## Completion checklist

- [ ] I can compute dot products and explain them geometrically.
- [ ] I understand L1 vs L2 vs L∞ norms.
- [ ] I can compute distances and interpret them.
- [ ] I can reason about matrix multiplication shapes.
- [ ] I can explain eigenvectors/eigenvalues as stable directions.
- [ ] I can explain SVD at a high level and why it is stable.
- [ ] Module tests pass.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| [01_PYTHON_CIENTIFICO](01_PYTHON_CIENTIFICO.md) | [00_INDICE](00_INDICE.md) | [03_CALCULO_MULTIVARIANTE](03_CALCULO_MULTIVARIANTE.md) |
