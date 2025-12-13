# 3D Visualization Guide for Gradient Descent

> Goal: build geometric intuition for optimization without relying on static images.

**Language:** [Español →](../VISUALIZACION_GRADIENT_DESCENT.md)

---

## What you should learn

- Why the gradient points “uphill”
- Why learning rate controls stability vs divergence
- How the trajectory changes with step count

---

## Recommended tool (provided)

Run the provided visualization (exports interactive HTML):

- `visualizations/viz_gradient_3d.py`

Examples:

```bash
python3 visualizations/viz_gradient_3d.py --lr 0.01 --steps 30 --html-out artifacts/gd_lr0_01.html
python3 visualizations/viz_gradient_3d.py --lr 1.0 --steps 30 --html-out artifacts/gd_lr1_0.html
```

Deliverable:

- 2 exports or screenshots (convergent vs divergent)
- 5-line explanation of what happened
