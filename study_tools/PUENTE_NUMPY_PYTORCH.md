# 🌉 Puente NumPy → PyTorch

> **Semana 24**: Toma tu clase NeuralNetwork hecha desde cero y reescríbela en PyTorch.
> Verás que tus 100 líneas se convierten en 5. Entenderás exactamente qué hace cada una.

---

## 🎯 Objetivo

Esta guía te conecta con la industria y las herramientas de cursos avanzados.
Al terminar, tendrás la "iluminación" de saber QUÉ hace PyTorch por ti.

---

## 📊 Tabla de Equivalencias NumPy ↔ PyTorch

### Operaciones Básicas

| Operación | NumPy | PyTorch |
|-----------|-------|---------|
| Crear array/tensor | `np.array([1, 2, 3])` | `torch.tensor([1, 2, 3])` |
| Zeros | `np.zeros((3, 4))` | `torch.zeros(3, 4)` |
| Ones | `np.ones((3, 4))` | `torch.ones(3, 4)` |
| Random normal | `np.random.randn(3, 4)` | `torch.randn(3, 4)` |
| Shape | `x.shape` | `x.shape` o `x.size()` |
| Reshape | `x.reshape(2, 6)` | `x.reshape(2, 6)` o `x.view(2, 6)` |
| Transponer | `x.T` | `x.T` o `x.t()` |
| Producto matricial | `A @ B` o `np.dot(A, B)` | `A @ B` o `torch.mm(A, B)` |
| Elemento a elemento | `A * B` | `A * B` |
| Suma | `np.sum(x, axis=0)` | `torch.sum(x, dim=0)` |
| Mean | `np.mean(x, axis=1)` | `torch.mean(x, dim=1)` |
| Max | `np.max(x)` | `torch.max(x)` |
| Argmax | `np.argmax(x, axis=1)` | `torch.argmax(x, dim=1)` |
| Concatenar | `np.concatenate([a, b])` | `torch.cat([a, b])` |
| Stack | `np.stack([a, b])` | `torch.stack([a, b])` |

### Funciones Matemáticas

| Operación | NumPy | PyTorch |
|-----------|-------|---------|
| Exponencial | `np.exp(x)` | `torch.exp(x)` |
| Logaritmo | `np.log(x)` | `torch.log(x)` |
| ReLU | `np.maximum(0, x)` | `torch.relu(x)` o `F.relu(x)` |
| Sigmoid | `1 / (1 + np.exp(-x))` | `torch.sigmoid(x)` |
| Tanh | `np.tanh(x)` | `torch.tanh(x)` |
| Softmax | manual | `F.softmax(x, dim=1)` |

### Conversión

| Dirección | Código |
|-----------|--------|
| NumPy → PyTorch | `torch.from_numpy(np_array)` |
| PyTorch → NumPy | `tensor.numpy()` (CPU) o `tensor.cpu().numpy()` (GPU) |
| PyTorch → Python | `tensor.item()` (escalar) |

---

## 🔄 Ejercicio de Traducción: Red Neuronal Completa

### Tu Código NumPy (100+ líneas)

```python
import numpy as np

class NeuralNetworkNumPy:
    """Red neuronal desde cero con NumPy."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Inicialización Xavier
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Cache para backward
        self.cache = {}
        
    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Capa 1
        self.cache['X'] = X
        self.cache['Z1'] = X @ self.W1 + self.b1
        self.cache['A1'] = self.relu(self.cache['Z1'])
        
        # Capa 2
        self.cache['Z2'] = self.cache['A1'] @ self.W2 + self.b2
        self.cache['A2'] = self.softmax(self.cache['Z2'])
        
        return self.cache['A2']
    
    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        m = y_true.shape[0]
        # Evitar log(0)
        log_probs = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-8)
        return np.mean(log_probs)
    
    def backward(self, y_true: np.ndarray) -> dict:
        m = y_true.shape[0]
        grads = {}
        
        # Gradiente de softmax + cross-entropy (simplificado)
        dZ2 = self.cache['A2'] - y_true  # (m, output_size)
        
        # Gradientes capa 2
        grads['dW2'] = (1/m) * self.cache['A1'].T @ dZ2
        grads['db2'] = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Propagar hacia atrás
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        
        # Gradientes capa 1
        grads['dW1'] = (1/m) * self.cache['X'].T @ dZ1
        grads['db1'] = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return grads
    
    def update_params(self, grads: dict, learning_rate: float):
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']
    
    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        # Forward
        y_pred = self.forward(X)
        loss = self.cross_entropy_loss(y_pred, y)
        
        # Backward
        grads = self.backward(y)
        
        # Update
        self.update_params(grads, learning_rate)
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
```

---

### Código Equivalente en PyTorch (15 líneas)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkPyTorch(nn.Module):
    """La misma red, pero PyTorch hace el trabajo pesado."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # W1, b1 incluidos
        self.fc2 = nn.Linear(hidden_size, output_size)  # W2, b2 incluidos
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # Capa 1 + ReLU
        x = self.fc2(x)          # Capa 2 (sin softmax, lo hace CrossEntropyLoss)
        return x
```

---

### Entrenamiento: NumPy vs PyTorch

#### NumPy (manual)

```python
# Crear modelo
model_np = NeuralNetworkNumPy(784, 128, 10)

# Training loop
for epoch in range(100):
    loss = model_np.train_step(X_train, y_train_onehot, learning_rate=0.01)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

#### PyTorch (automático)

```python
# Crear modelo
model_pt = NeuralNetworkPyTorch(784, 128, 10)

# Loss y optimizer (PyTorch los maneja)
criterion = nn.CrossEntropyLoss()  # Incluye softmax!
optimizer = torch.optim.SGD(model_pt.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()           # Limpiar gradientes anteriores
    
    outputs = model_pt(X_train_t)   # Forward pass
    loss = criterion(outputs, y_train_t)  # Calcular loss
    
    loss.backward()                 # Backward pass (automático!)
    optimizer.step()                # Actualizar pesos
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 🔍 Mapeo Línea por Línea

| Lo que TÚ escribiste (NumPy) | Lo que PyTorch hace por ti |
|------------------------------|---------------------------|
| `self.W1 = np.random.randn(...) * np.sqrt(2.0/n)` | `nn.Linear` inicializa automáticamente (Kaiming) |
| `self.b1 = np.zeros(...)` | `nn.Linear` incluye bias automáticamente |
| `self.cache['Z1'] = X @ self.W1 + self.b1` | `self.fc1(x)` |
| `self.relu(z)` | `F.relu(x)` |
| `self.softmax(z)` | Incluido en `nn.CrossEntropyLoss` |
| Todo tu `backward()` | `loss.backward()` (Autograd!) |
| `self.W1 -= lr * grads['dW1']` | `optimizer.step()` |
| `self.cache = {}` | PyTorch mantiene el grafo automáticamente |

---

## 🧪 Verificación: Mismo Resultado

```python
import numpy as np
import torch
import torch.nn as nn

def verificar_equivalencia():
    """Verifica que ambas implementaciones dan el mismo resultado."""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Datos de prueba
    X = np.random.randn(10, 784).astype(np.float32)
    y = np.eye(10)[np.random.randint(0, 10, 10)]  # One-hot
    
    # Modelo NumPy
    model_np = NeuralNetworkNumPy(784, 128, 10)
    
    # Modelo PyTorch (copiar pesos del modelo NumPy)
    model_pt = NeuralNetworkPyTorch(784, 128, 10)
    with torch.no_grad():
        model_pt.fc1.weight.copy_(torch.from_numpy(model_np.W1.T))
        model_pt.fc1.bias.copy_(torch.from_numpy(model_np.b1.flatten()))
        model_pt.fc2.weight.copy_(torch.from_numpy(model_np.W2.T))
        model_pt.fc2.bias.copy_(torch.from_numpy(model_np.b2.flatten()))
    
    # Forward pass NumPy
    output_np = model_np.forward(X)
    
    # Forward pass PyTorch
    X_t = torch.from_numpy(X)
    with torch.no_grad():
        output_pt = torch.softmax(model_pt(X_t), dim=1).numpy()
    
    # Comparar
    diff = np.abs(output_np - output_pt).max()
    print(f"Diferencia máxima en outputs: {diff:.2e}")
    assert diff < 1e-5, "¡Los outputs no coinciden!"
    print("✅ ¡Verificación exitosa! Ambos modelos son equivalentes.")

verificar_equivalencia()
```

---

## 🚀 Ventajas de PyTorch que Ahora Entiendes

| Feature | Qué hace | Por qué es poderoso |
|---------|----------|---------------------|
| **Autograd** | Calcula gradientes automáticamente | No más errores en backprop manual |
| **nn.Module** | Organiza parámetros | `model.parameters()` los encuentra todos |
| **Optimizers** | SGD, Adam, etc. listo para usar | No reimplementar momentum, weight decay |
| **GPU** | `.to('cuda')` | Entrenamiento 10-100x más rápido |
| **DataLoader** | Batching, shuffling automático | No más `X[i:i+batch_size]` |
| **Loss Functions** | CrossEntropy, MSE, etc. | Numéricamente estables |

---

## 📝 Ejercicio Final: MNIST en PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Cargar datos (1 línea vs tu código de preprocesamiento)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Definir modelo (tu clase de 100 líneas → 8 líneas)
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 3. Entrenar (tu training loop simplificado)
model = MNISTNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

# 4. Evaluar
model.eval()
correct = 0
with torch.no_grad():
    for data, target in train_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()

print(f"Accuracy: {100*correct/len(train_data):.2f}%")
```

---

## ✅ Checklist de "Iluminación"

Después de este ejercicio, deberías poder responder:

- [ ] ¿Qué hace `nn.Linear` internamente?
- [ ] ¿Por qué `nn.CrossEntropyLoss` no necesita softmax explícito?
- [ ] ¿Qué hace `loss.backward()` exactamente?
- [ ] ¿Por qué necesitamos `optimizer.zero_grad()`?
- [ ] ¿Qué es `model.parameters()` y por qué funciona?
- [ ] ¿Cómo movería este modelo a GPU?

---

## 🎓 Conclusión

> "Después de implementar todo desde cero, PyTorch no es magia. Es automatización de lo que ya sabes hacer."

Ahora estás listo para:
1. Cursos avanzados de la maestría que usan PyTorch/TensorFlow
2. Entender papers que usan frameworks
3. Debuggear modelos porque sabes QUÉ pasa por debajo
4. Extender PyTorch con operaciones custom si es necesario
