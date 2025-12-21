import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def make_xor(n: int = 400, seed: int = 42, noise: float = 0.15):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = (X[:, 0] * X[:, 1] < 0).astype(np.int64)
    X = X + rng.normal(0.0, noise, size=X.shape)
    return X.astype(np.float32), y


st.set_page_config(page_title="PyTorch Playground", layout="centered")

st.title("Playground: Red neuronal entrenando en vivo (PyTorch + Streamlit)")
st.write(
    "Entrena una MLP pequeña en un dataset tipo XOR y observa la pérdida y la frontera de decisión."
)

try:
    import torch
    import torch.nn as nn
except Exception:
    st.error(
        "PyTorch no está instalado. Instala el extra pytorch o ejecuta: pip install torch torchvision"
    )
    st.stop()

seed = st.number_input("Semilla", value=42, step=1)
hidden = st.slider("Hidden units", min_value=2, max_value=64, value=16, step=2)
lr = st.slider(
    "Learning rate",
    min_value=1e-4,
    max_value=1e-1,
    value=1e-2,
    step=1e-4,
    format="%.4f",
)
epochs = st.slider("Epochs", min_value=50, max_value=2000, value=400, step=50)

X_np, y_np = make_xor(seed=int(seed))
X = torch.tensor(X_np)
y = torch.tensor(y_np)

model = nn.Sequential(
    nn.Linear(2, int(hidden)),
    nn.Tanh(),
    nn.Linear(int(hidden), 2),
)

opt = torch.optim.Adam(model.parameters(), lr=float(lr))
loss_fn = nn.CrossEntropyLoss()

losses = []

for _ in range(int(epochs)):
    logits = model(X)
    loss = loss_fn(logits, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss.detach().cpu().item()))

# Accuracy
with torch.no_grad():
    pred = torch.argmax(model(X), dim=1)
    acc = float((pred == y).float().mean().cpu().item())

# Plot loss
fig1, ax1 = plt.subplots(figsize=(7, 3))
ax1.plot(losses)
ax1.set_title(f"Loss (acc={acc:.3f})")
ax1.set_xlabel("Step")
ax1.set_ylabel("Cross-Entropy")
ax1.grid(True, alpha=0.2)

# Decision boundary
x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

with torch.no_grad():
    logits = model(torch.tensor(grid))
    proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.contourf(xx, yy, proba, levels=30, cmap="RdBu", alpha=0.35)
ax2.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=2)
ax2.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], s=12, label="Clase 0")
ax2.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], s=12, label="Clase 1")
ax2.set_title("Frontera de decisión")
ax2.legend()
ax2.grid(True, alpha=0.2)

st.pyplot(fig1)
st.pyplot(fig2)

st.info(
    "Ejecuta: streamlit run interactive_labs/m07_deep_learning/pytorch_training_playground_app.py"
)
