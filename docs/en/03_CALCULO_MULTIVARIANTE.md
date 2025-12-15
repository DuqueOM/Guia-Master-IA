# Module 03 - Multivariate Calculus for Deep Learning

> **Goal:** master derivatives, gradients, and the Chain Rule to understand backpropagation.
> **Phase:** 1 - Math Foundations | **Weeks 6–8**
> **Prerequisites:** Module 02 (Linear Algebra)

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

## 🎯 Topic-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 15–30 min per exercise.
- **Minimum success:** your solution must pass the `assert` checks.

---

### Exercise 3.1: Numerical derivative (finite differences) vs analytic derivative

#### Prompt

1) **Basic**

- Implement the central difference derivative: `f'(x) ≈ (f(x+h)-f(x-h))/(2h)`.

2) **Intermediate**

- For `f(x) = x^3 + 2x`, implement the analytic derivative and compare at multiple points.

3) **Advanced**

- Try `h=1e-2, 1e-4, 1e-6` and verify the error does not blow up.

#### Solution

```python
import numpy as np

def num_derivative_central(f, x: float, h: float = 1e-6) -> float:
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def f(x: float) -> float:
    return x**3 + 2.0 * x


def f_prime(x: float) -> float:
    return 3.0 * x**2 + 2.0


xs = [-2.0, -0.5, 0.0, 1.0, 3.0]
for x in xs:
    approx = num_derivative_central(f, x, h=1e-6)
    exact = f_prime(x)
    assert np.isclose(approx, exact, rtol=1e-6, atol=1e-6)


x0 = 1.234
errs = []
for h in [1e-2, 1e-4, 1e-6]:
    approx = num_derivative_central(f, x0, h=h)
    errs.append(abs(approx - f_prime(x0)))
assert errs[1] <= errs[0] + 1e-6
```

---

### Exercise 3.2: Partial derivatives and gradient (2D)

#### Prompt

Let `f(x, y) = x^2 y + sin(y)`.

1) **Basic**

- Derive analytically `∂f/∂x` and `∂f/∂y`.

2) **Intermediate**

- Implement the gradient `∇f(x,y)` and evaluate it at a point.

3) **Advanced**

- Verify with a numerical gradient (central differences).

#### Solution

```python
import numpy as np

def f_xy(x: float, y: float) -> float:
    return x**2 * y + np.sin(y)


def grad_f_xy(x: float, y: float) -> np.ndarray:
    dfdx = 2.0 * x * y
    dfdy = x**2 + np.cos(y)
    return np.array([dfdx, dfdy], dtype=float)


def num_grad_2d(f, x: float, y: float, h: float = 1e-6) -> np.ndarray:
    dfdx = (f(x + h, y) - f(x - h, y)) / (2.0 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2.0 * h)
    return np.array([dfdx, dfdy], dtype=float)


x0, y0 = 1.2, -0.7
g_anal = grad_f_xy(x0, y0)
g_num = num_grad_2d(f_xy, x0, y0)
assert np.allclose(g_anal, g_num, rtol=1e-5, atol=1e-6)
```

---

### Exercise 3.3: Directional derivative (intuition: the gradient rules)

#### Prompt

1) **Basic**

- For `f(x,y)=x^2 y + sin(y)`, compute `∇f(x0,y0)`.

2) **Intermediate**

- Given a unit direction vector `u`, compute the directional derivative `D_u f = ∇f · u`.

3) **Advanced**

- Verify `D_u f` numerically with finite differences along `p(t)=p0 + t u`.

#### Solution

```python
import numpy as np

def f_xy(x: float, y: float) -> float:
    return x**2 * y + np.sin(y)


def grad_f_xy(x: float, y: float) -> np.ndarray:
    return np.array([2.0 * x * y, x**2 + np.cos(y)], dtype=float)


x0, y0 = 0.5, 1.0
g = grad_f_xy(x0, y0)

u = np.array([3.0, 4.0], dtype=float)
u = u / np.linalg.norm(u)

dir_anal = float(np.dot(g, u))

h = 1e-6
f_plus = f_xy(x0 + h * u[0], y0 + h * u[1])
f_minus = f_xy(x0 - h * u[0], y0 - h * u[1])
dir_num = float((f_plus - f_minus) / (2.0 * h))

assert np.isclose(dir_anal, dir_num, rtol=1e-5, atol=1e-6)
```

---

### Exercise 3.4: Jacobian (vector-valued function)

#### Prompt

Let `g(x1,x2) = [x1^2 + x2, sin(x1 x2)]`.

1) **Basic**

- Write the Jacobian `J` (2x2) by hand.

2) **Intermediate**

- Implement `J_analytical(x)`.

3) **Advanced**

- Verify with a numerical Jacobian (central differences).

#### Solution

```python
import numpy as np

def g(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    return np.array([x1**2 + x2, np.sin(x1 * x2)], dtype=float)


def J_analytical(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    dg1_dx1 = 2.0 * x1
    dg1_dx2 = 1.0
    dg2_dx1 = np.cos(x1 * x2) * x2
    dg2_dx2 = np.cos(x1 * x2) * x1
    return np.array([[dg1_dx1, dg1_dx2], [dg2_dx1, dg2_dx2]], dtype=float)


def J_numeric(g, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    x = x.astype(float)
    m = g(x).shape[0]
    n = x.shape[0]
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        J[:, j] = (g(x + h * e) - g(x - h * e)) / (2.0 * h)
    return J


x0 = np.array([0.7, -1.1])
Ja = J_analytical(x0)
Jn = J_numeric(g, x0)
assert np.allclose(Ja, Jn, rtol=1e-5, atol=1e-6)
```

---

### Exercise 3.5: Hessian (local curvature) + convexity

#### Prompt

Let `f(x1,x2) = x1^2 + 2 x2^2`.

1) **Basic**

- Compute the Hessian `H`.

2) **Intermediate**

- Verify `H` is symmetric.

3) **Advanced**

- Verify `H` is positive definite (eigenvalues > 0).

#### Solution

```python
import numpy as np

H = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=float)
assert np.allclose(H, H.T)

eigvals = np.linalg.eigvals(H)
assert np.all(eigvals > 0)
```

---

### Exercise 3.6: Gradient Descent in 1D (convergence)

#### Prompt

Minimize `f(x) = (x - 3)^2` with Gradient Descent.

1) **Basic**

- Implement the update rule: `x <- x - α f'(x)`.

2) **Intermediate**

- Track `x_t` and `f(x_t)`.

3) **Advanced**

- Use a stopping criterion like `|grad| < tol`.

#### Solution

```python
import numpy as np

def f(x: float) -> float:
    return (x - 3.0) ** 2


def grad_f(x: float) -> float:
    return 2.0 * (x - 3.0)


x = 10.0
alpha = 0.1
history = []
for _ in range(200):
    g = grad_f(x)
    history.append((x, f(x)))
    if abs(g) < 1e-8:
        break
    x = x - alpha * g

assert abs(x - 3.0) < 1e-4
assert history[-1][1] <= history[0][1]
```

---

### Exercise 3.7: Learning rate effect (stability)

#### Prompt

Minimize `f(x)=x^2` with Gradient Descent from `x0=1`.

1) **Basic**

- Derive: `x_{t+1} = (1 - 2α) x_t`.

2) **Intermediate**

- Try `α=0.25` and verify `|x_t|` decreases.

3) **Advanced**

- Try `α=1.1` and verify divergence (`|x_t|` increases).

#### Solution

```python
import numpy as np

def run_gd_x2(alpha: float, steps: int = 10) -> np.ndarray:
    x = 1.0
    xs = [x]
    for _ in range(steps):
        grad = 2.0 * x
        x = x - alpha * grad
        xs.append(x)
    return np.array(xs)


xs_good = run_gd_x2(alpha=0.25, steps=10)
assert abs(xs_good[-1]) < abs(xs_good[0])

xs_bad = run_gd_x2(alpha=1.1, steps=10)
assert abs(xs_bad[-1]) > abs(xs_bad[0])
```

---

### Exercise 3.8: Gradient checking (vector) + relative error

#### Prompt

1) **Basic**

- Implement a numerical gradient (central differences) for `f(w)`.

2) **Intermediate**

- Use `f(w)=∑ w_i^3` with analytic gradient `3 w_i^2`.

3) **Advanced**

- Compute the relative error `||g_num - g_anal|| / (||g_num|| + ||g_anal|| + eps)`.

#### Solution

```python
import numpy as np

def f(w: np.ndarray) -> float:
    return float(np.sum(w ** 3))


def grad_analytical(w: np.ndarray) -> np.ndarray:
    return 3.0 * (w ** 2)


def grad_numeric(f, w: np.ndarray, h: float = 1e-6) -> np.ndarray:
    w = w.astype(float)
    g = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = 1.0
        g[i] = (f(w + h * e) - f(w - h * e)) / (2.0 * h)
    return g


np.random.seed(0)
w = np.random.randn(5)
g_a = grad_analytical(w)
g_n = grad_numeric(f, w)

eps = 1e-12
rel_err = np.linalg.norm(g_n - g_a) / (np.linalg.norm(g_n) + np.linalg.norm(g_a) + eps)
assert rel_err < 1e-7
```

---

### Exercise 3.9: Chain Rule (neuron + MSE) + numerical verification

#### Prompt

A single neuron:

- `z = w·x + b`
- `ŷ = σ(z)`
- `L = (ŷ - y)^2`

1) **Basic**

- Derive `dL/dz` using the Chain Rule.

2) **Intermediate**

- Derive `dL/dw` and `dL/db`.

3) **Advanced**

- Verify with central differences (gradient checking).

#### Solution

```python
import numpy as np

def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def loss_mse(y_hat: float, y: float) -> float:
    return float((y_hat - y) ** 2)


def forward(w: np.ndarray, b: float, x: np.ndarray, y: float) -> float:
    z = float(np.dot(w, x) + b)
    y_hat = sigmoid(z)
    return loss_mse(y_hat, y)


def grads_analytical(w: np.ndarray, b: float, x: np.ndarray, y: float):
    z = float(np.dot(w, x) + b)
    y_hat = sigmoid(z)

    dL_dyhat = 2.0 * (y_hat - y)
    dyhat_dz = y_hat * (1.0 - y_hat)
    dL_dz = dL_dyhat * dyhat_dz

    dL_dw = dL_dz * x
    dL_db = dL_dz
    return dL_dw.astype(float), float(dL_db)


def grads_numeric(w: np.ndarray, b: float, x: np.ndarray, y: float, h: float = 1e-6):
    gw = np.zeros_like(w, dtype=float)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = 1.0
        gw[i] = (forward(w + h * e, b, x, y) - forward(w - h * e, b, x, y)) / (2.0 * h)

    gb = (forward(w, b + h, x, y) - forward(w, b - h, x, y)) / (2.0 * h)
    return gw, float(gb)


np.random.seed(1)
w = np.random.randn(3)
b = 0.1
x = np.random.randn(3)
y = 1.0

gw_a, gb_a = grads_analytical(w, b, x, y)
gw_n, gb_n = grads_numeric(w, b, x, y)

assert np.allclose(gw_a, gw_n, rtol=1e-5, atol=1e-6)
assert np.isclose(gb_a, gb_n, rtol=1e-5, atol=1e-6)
```

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
