from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

Criterion = Literal["gini", "entropy"]


def _gini(y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


def _entropy(y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def _impurity(y: np.ndarray, criterion: Criterion) -> float:
    if criterion == "gini":
        return _gini(y)
    if criterion == "entropy":
        return _entropy(y)
    raise ValueError(f"Unknown criterion: {criterion}")


def _information_gain(
    y_parent: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    criterion: Criterion,
) -> float:
    n = y_parent.size
    if n == 0:
        return 0.0

    parent_impurity = _impurity(y_parent, criterion)

    n_left = y_left.size
    n_right = y_right.size
    if n_left == 0 or n_right == 0:
        return 0.0

    child_impurity = (n_left / n) * _impurity(y_left, criterion) + (
        n_right / n
    ) * _impurity(y_right, criterion)
    return float(parent_impurity - child_impurity)


@dataclass
class _Node:
    feature_index: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    value: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: Criterion = "gini",
        n_thresholds: int = 32,
        random_state: int = 42,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.criterion = criterion
        self.n_thresholds = int(n_thresholds)
        self.random_state = int(random_state)

        self.root_: _Node | None = None
        self.n_classes_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        classes = np.unique(y)
        self.n_classes_ = int(classes.size)

        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        return np.array([self._predict_one(row, self.root_) for row in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        cur = node
        while not cur.is_leaf:
            if cur.feature_index is None or cur.threshold is None:
                raise RuntimeError("Corrupted tree node")
            if x[cur.feature_index] <= cur.threshold:
                if cur.left is None:
                    raise RuntimeError("Missing left child")
                cur = cur.left
            else:
                if cur.right is None:
                    raise RuntimeError("Missing right child")
                cur = cur.right
        if cur.value is None:
            raise RuntimeError("Leaf node missing value")
        return int(cur.value)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples, _ = X.shape

        y_counts = Counter(y.tolist())
        majority_class = int(max(y_counts.items(), key=lambda kv: kv[1])[0])

        if depth >= self.max_depth:
            return _Node(value=majority_class)

        if n_samples < self.min_samples_split:
            return _Node(value=majority_class)

        unique_classes = np.unique(y)
        if unique_classes.size == 1:
            return _Node(value=int(unique_classes[0]))

        split = self._best_split(X, y)
        if split is None:
            return _Node(value=majority_class)

        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth=depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth=depth + 1)

        return _Node(
            feature_index=feature_index,
            threshold=float(threshold),
            left=left,
            right=right,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        best_gain = 0.0
        best: tuple[int, float] | None = None

        for j in range(n_features):
            col = X[:, j]
            unique_vals = np.unique(col)
            if unique_vals.size < 2:
                continue

            if unique_vals.size <= self.n_thresholds + 1:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            else:
                qs = np.linspace(0.02, 0.98, num=self.n_thresholds)
                thresholds = np.quantile(unique_vals, qs)
                thresholds = np.unique(thresholds)
                thresholds = thresholds.astype(float)
                rng.shuffle(thresholds)

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                gain = _information_gain(y, y[left_mask], y[right_mask], self.criterion)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (j, float(thr))

        return best


def _make_toy_dataset(n: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] + 0.25 * rng.normal(size=n) > 0.75).astype(int)
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decision Tree from scratch (ID3/CART-style splits, no gradients)."
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--criterion", choices=["gini", "entropy"], default="gini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = _make_toy_dataset(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    tree = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        criterion=args.criterion,
        random_state=args.seed,
    )
    tree.fit(X_train, y_train)

    acc_train = tree.score(X_train, y_train)
    acc_test = tree.score(X_test, y_test)

    print("=== Decision Tree from scratch ===")
    print(f"criterion: {args.criterion}")
    print(f"max_depth: {args.max_depth}")
    print(f"train accuracy: {acc_train:.3f}")
    print(f"test accuracy:  {acc_test:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
