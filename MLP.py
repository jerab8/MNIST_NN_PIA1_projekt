import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/home/adamb/Stažené/winequality-white.csv", sep=";")

print(data.shape)   # mělo by být (4898, 12)
print(data.columns) # ukáže názvy sloupců
print(data.head())  # prvních 5 řádků

X = data.drop("quality", axis=1).values   # vstupy
y = data["quality"].values                # cílový sloupec

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.T
X_test = X_test.T

import numpy as np

class MLP:
    def __init__(self, layers, hidden_act="relu", out_act="identity", lr=1e-2):
        self.layers = layers
        self.hidden_act = hidden_act
        self.out_act = out_act
        self.lr = lr
        self.params = self.init_params(layers)

    # -----------------------------
    # Initialization utilities
    # -----------------------------
    def he_init(self, fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)

    def xavier_init(self, fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(1.0 / fan_in)

    def init_params(self, layers):
        params = {}
        for l in range(1, len(layers)):
            fan_in, fan_out = layers[l-1], layers[l]
            if self.hidden_act in ("relu", "leaky_relu"):
                W = self.he_init(fan_in, fan_out)
            else:
                W = self.xavier_init(fan_in, fan_out)
            b = np.zeros((fan_out, 1))
            params[f"W{l}"], params[f"b{l}"] = W, b
        return params

    # -----------------------------
    # Activations
    # -----------------------------
    def activation(self, z, kind="relu", deriv=False, axis=1, alpha=0.01, eps=1e-12):
        kind = kind.lower()
        if kind == "relu":
            y = np.maximum(0, z)
            return (y, np.where(z > 0, 1.0, 0.0)) if deriv else y
        elif kind == "leaky_relu":
            y = np.where(z > 0, z, alpha * z)
            return (y, np.where(z > 0, 1.0, alpha)) if deriv else y
        elif kind == "sigmoid":
            y = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            return (y, y * (1 - y)) if deriv else y
        elif kind == "tanh":
            y = np.tanh(z)
            return (y, 1 - y**2) if deriv else y
        elif kind == "identity":
            return (z, np.ones_like(z)) if deriv else z
        elif kind == "softmax":
            z_max = np.max(z, axis=axis, keepdims=True)
            e = np.exp(z - z_max)
            y = e / (np.sum(e, axis=axis, keepdims=True) + eps)
            if not deriv:
                return y
            def jvp(g):
                dot = np.sum(g * y, axis=axis, keepdims=True)
                return (g - dot) * y
            return y, jvp
        else:
            raise ValueError(f"Unknown activation: {kind}")

    # -----------------------------
    # Loss
    # -----------------------------
    def mse_loss(self, y_pred, y_true, reduction="mean"):
        diff = y_pred - y_true
        se = 0.5 * diff**2
        if diff.ndim <= 1:
            per_sample = se
        else:
            per_sample = np.sum(se, axis=tuple(range(1, diff.ndim)))
        if reduction == "mean":
            N = y_pred.shape[1] if y_pred.ndim >= 2 else y_pred.shape[0]
            return np.sum(per_sample) / max(N,1), diff / max(N,1)
        elif reduction == "sum":
            return np.sum(per_sample), diff
        else:
            return per_sample, diff

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, X):
        A = X
        caches = []
        L = len(self.params) // 2
        for l in range(1, L):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            Z = W @ A + b
            A = self.activation(Z, self.hidden_act)
            caches.append((Z, A, W, b, self.hidden_act))
        W, b = self.params[f"W{L}"], self.params[f"b{L}"]
        Z = W @ A + b
        A = self.activation(Z, self.out_act)
        caches.append((Z, A, W, b, self.out_act))
        return A, caches

    # -----------------------------
    # Backward
    # -----------------------------
    def backward(self, caches, dAL):
        grads = {}
        dA = dAL
        for l in reversed(range(len(caches))):
            Z, A, W, b, kind = caches[l]
            A_prev = caches[l-1][1] if l > 0 else None
            if kind in ("relu","leaky_relu","identity"):
                _, d = self.activation(Z, kind, deriv=True)
                dZ = dA * d
            elif kind in ("sigmoid","tanh"):
                Y, d = self.activation(Z, kind, deriv=True)
                dZ = dA * d
            elif kind == "softmax":
                jvp = self.activation(Z, "softmax", deriv=True)[1]
                dZ = jvp(dA)
            m = A_prev.shape[1] if A_prev is not None else dZ.shape[1]
            dW = (dZ @ A_prev.T) / m if A_prev is not None else None
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = W.T @ dZ
            grads[f"dW{l+1}"], grads[f"db{l+1}"] = dW, db
        return grads

    # -----------------------------
    # Update
    # -----------------------------
    def sgd_update(self, grads):
        L = len(self.params) // 2
        for l in range(1, L+1):
            if grads[f"dW{l}"] is not None:
                self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    # -----------------------------
    # Training step
    # -----------------------------

    
    def train_step(self, X_batch, Y_batch):
        Y_pred, caches = self.forward(X_batch)
        loss, dY = self.mse_loss(Y_pred, Y_batch, reduction="mean")
        grads = self.backward(caches, dY)
        self.sgd_update(grads)
        return loss


# -----------------------------
# Použití
# -----------------------------
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False)  # novější verze sklearn
y_train_oh = enc.fit_transform(y_train.reshape(-1, 1)).T
y_test_oh = enc.transform(y_test.reshape(-1, 1)).T

def accuracy(y_pred, y_true, threshold=0.5):
    # pokud používáš sigmoid na výstupu
    preds = (y_pred.flatten() > threshold).astype(int)
    true = y_true.flatten().astype(int)
    return np.mean(preds == true)

n_classes = data["quality"].nunique()  # 7

mlp = MLP(
    layers=[X_train.shape[0], 32, 16, n_classes],
    hidden_act="relu",
    out_act="softmax",
    lr=0.01
)

for epoch in range(100):
    loss = mlp.train_step(X_train, y_train_oh)
    if epoch % 10 == 0:
        y_pred, _ = mlp.forward(X_train)
        acc = accuracy(y_pred, y_train_oh)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")


y_pred_test, _ = mlp.forward(X_test)
test_acc = accuracy(y_pred_test, y_test_oh)
print(f"Final Test Accuracy: {test_acc:.4f}")

# vezmeme první vzorek z testovací množiny
x_sample = X_test[:, 0].reshape(-1, 1)   # vstup má tvar [n_features, 1]
y_true = y_test_oh[0]                       # skutečná třída

# predikce modelem
y_pred, _ = mlp.forward(x_sample)

print("True label:", y_true)
print("Predicted value (raw):", y_pred.flatten()[0])
print("Predicted class:", int(y_pred.flatten()[0] > 0.5))

