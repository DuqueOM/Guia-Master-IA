# Module 03 - Multivariate Calculus for Deep Learning

> **Goal:** master derivatives, gradients, and the Chain Rule to understand backpropagation.
> **Phase:** 1 - Math Foundations | **Weeks 6–8**
> **Prerequisites:** Module 02 (Linear Algebra)

**Language:** English | [Español →](../03_CALCULO_MULTIVARIANTE.md)

---

<a id="m03-0"></a>

## How to use this module (0→100 mode)

**Purpose:** be able to do 3 things without “faith”:

- derive gradients for common losses (MSE, BCE)
- implement and debug optimization (Gradient Descent)
- understand backprop as Chain Rule on a computation graph

### Learning outcomes (measurable)

By the end of this module you can:

- Compute derivatives and partial derivatives (by hand + numerical verification).
- Apply gradients as the direction of steepest ascent (and descent for minimization).
- Implement Gradient Descent with reasonable convergence criteria.
- Explain the Chain Rule and use it on composed functions.
- Validate your math using gradient checking (small relative error).

Quick references (Spanish source, stable anchors):

- [Glossary: Derivative](GLOSARIO.md#derivative)
- [Glossary: Gradient](GLOSARIO.md#gradient)
- [Glossary: Gradient Descent](GLOSARIO.md#gradient-descent)
- [Glossary: Chain Rule](GLOSARIO.md#chain-rule)
- [Resources](RECURSOS.md)

### Integration with v4/v5

- Optimization visualization guide: `study_tools/VISUALIZACION_GRADIENT_DESCENT.md`
- Exam drills: `study_tools/SIMULACRO_EXAMEN_TEORICO.md`
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M03` in `rubrica.csv`)
- Protocols:
  - [Plan v4](PLAN_V4_ESTRATEGICO.md)
  - [Plan v5](PLAN_V5_ESTRATEGICO.md)

---

## Recommended resources (when to use them)

| Priority | Resource | When | Why |
|---|---|---|---|
| Required | `study_tools/VISUALIZACION_GRADIENT_DESCENT.md` | When tuning `learning_rate` and stopping criteria | Build intuition for convergence/divergence |
| Recommended | `../visualizations/viz_gradient_3d.py` | Week 7 | Generate an interactive 3D HTML (surface + trajectory) |
| Optional | 3Blue1Brown: Calculus | Before Chain Rule | Visual intuition for derivatives and composition |
| Recommended | Mathematics for ML: Multivariate Calculus | Transition to gradients/partials | Structured practice |

---

## Module content (Weeks 6–8)

- **Week 6:** derivatives + partial derivatives + numerical differentiation.
- **Week 7:** gradients + Gradient Descent (stability, learning rate intuition).
- **Week 8:** Chain Rule + preparation for manual backprop in Module 07.

Core invariant:

- If you cannot explain `∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w`, you will not be able to debug backprop.

---

## Deliverables

### Deliverable A: `gradient_descent_demo.py`

**Goal:** implement Gradient Descent from scratch and visualize behavior on multiple functions.

Minimum expectations:

- correct update rule
- a convergence criterion
- plots showing trajectory/convergence

### Deliverable B (required v3.3): `grad_check.py`

**Goal:** gradient checking via finite differences (central difference).

Minimum expectations:

- compare analytic vs numeric gradients
- report relative error
- small enough error to trust your derivatives

---

## Completion checklist (v3.3)

### Knowledge

- [ ] I can compute derivatives of common functions (polynomials, exp, log).
- [ ] I can compute partial derivatives.
- [ ] I can explain the gradient as a direction (geometry intuition).
- [ ] I can implement Gradient Descent.
- [ ] I understand how learning rate impacts stability.
- [ ] I understand Chain Rule and how it is used in backprop.
- [ ] I can derive `∂L/∂w` for a simple neuron.

### Deliverables v3.3

- [ ] `gradient_descent_demo.py` works.
- [ ] `grad_check.py` implemented and tests pass.
- [ ] I validated my derivatives for sigmoid, MSE, and a linear layer.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| [02_ALGEBRA_LINEAL_ML](02_ALGEBRA_LINEAL_ML.md) | [00_INDICE](00_INDICE.md) | [04_PROBABILIDAD_ML](04_PROBABILIDAD_ML.md) |
