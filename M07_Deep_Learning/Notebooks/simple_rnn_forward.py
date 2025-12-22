"""
Notebook M07: RNN Forward Pass desde Cero
==========================================

Módulo 7 - Semana 17: Deep Learning
Curso Alineado: CSCA 5642 - Deep Learning

Objetivos:
1. Implementar el forward pass de una RNN vanilla desde cero
2. Entender la propagación de estado oculto a través del tiempo
3. Conectar con fundamentos de álgebra lineal y cálculo

💡 PUENTES TEÓRICO-PRÁCTICOS:
-----------------------------
- M02 (Álgebra Lineal): La RNN aplica transformaciones lineales en cada paso:
  h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
  Cada @ es una multiplicación matriz-vector que viste en M02.

- M03 (Cálculo): El gradiente se propaga hacia atrás en el tiempo (BPTT).
  La regla de la cadena se aplica a través de todos los timesteps:
  ∂L/∂W = Σ_t ∂L/∂h_t × ∂h_t/∂W

- M04 (Probabilidad): En secuencias, modelamos P(x_t | x_{1:t-1}) condicionando
  en el estado oculto h_t que resume toda la historia.

⚠️ PROBLEMA DEL GRADIENTE EVANESCENTE:
  Como ∂h_t/∂h_{t-k} = Π W_hh × diag(1 - tanh²), si ||W_hh|| < 1,
  los gradientes decaen exponencialmente → LSTM/GRU resuelven esto.

Ejecutar: python simple_rnn_forward.py --batch 4 --time 6 --hidden 16
"""

from __future__ import annotations

import argparse

import numpy as np


def rnn_forward(
    x: np.ndarray,
    wxh: np.ndarray,
    whh: np.ndarray,
    b: np.ndarray,
    h0: np.ndarray,
    why: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch, time, d_in = x.shape
    d_hidden = h0.shape[1]
    d_out = c.shape[0]

    assert wxh.shape == (d_in, d_hidden)
    assert whh.shape == (d_hidden, d_hidden)
    assert b.shape == (d_hidden,)
    assert why.shape == (d_hidden, d_out)
    assert c.shape == (d_out,)

    hs = np.zeros((batch, time, d_hidden), dtype=np.float64)
    ys = np.zeros((batch, time, d_out), dtype=np.float64)

    h_t = h0
    for t in range(time):
        x_t = x[:, t, :]
        h_t = np.tanh(x_t @ wxh + h_t @ whh + b)
        y_t = h_t @ why + c
        hs[:, t, :] = h_t
        ys[:, t, :] = y_t

    return ys, hs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--time", type=int, default=6)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--out", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))

    x = rng.standard_normal((int(args.batch), int(args.time), int(args.features)))

    wxh = rng.standard_normal((int(args.features), int(args.hidden))) * 0.2
    whh = rng.standard_normal((int(args.hidden), int(args.hidden))) * 0.2
    b = np.zeros((int(args.hidden),), dtype=np.float64)

    why = rng.standard_normal((int(args.hidden), int(args.out))) * 0.2
    c = np.zeros((int(args.out),), dtype=np.float64)

    h0 = np.zeros((int(args.batch), int(args.hidden)), dtype=np.float64)

    y, h = rnn_forward(x, wxh, whh, b, h0, why, c)

    print(f"x.shape={x.shape}  (batch,time,features)")
    print(f"h.shape={h.shape}  (batch,time,hidden)")
    print(f"y.shape={y.shape}  (batch,time,out)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
