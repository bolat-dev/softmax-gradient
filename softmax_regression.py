import numpy as np


def softmax(z):
    # Subtract the row max for numerical stability
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# 9 samples, 2 features, 3 classes
X = np.array([
    [0.8, 0.9], [0.7, 0.8], [0.9, 0.85],      # class 0
    [0.4, 0.2], [0.3, 0.15], [0.5, 0.25],     # class 1
    [0.6, 0.7], [0.65, 0.75], [0.55, 0.65],   # class 2
])

# One-hot labels, shape (9, 3)
y = np.repeat(np.eye(3), 3, axis=0)

# Weights (features x classes) and bias
W = np.array([
    [0.2, 0.3, 0.5],
    [0.6, 0.4, 0.1],
])
b = np.zeros(3)

lr = 1.0
N = X.shape[0]

for i in range(1000):
    p = softmax(X @ W + b)          # predicted probabilities (y_hat)
    dW = X.T @ (p - y) / N          # dL/dW = (1/N) X^T (y_hat - y)
    db = (p - y).sum(axis=0) / N    # dL/db = (1/N) sum(y_hat - y)
    W -= lr * dW
    b -= lr * db

# Final evaluation
p = softmax(X @ W + b)
loss = -np.sum(y * np.log(p)) / N   # mean cross-entropy loss

np.set_printoptions(suppress=True, precision=4)
print(p)
print(y)
print("loss:", round(loss, 4))
print("accuracy:", (p.argmax(axis=1) == y.argmax(axis=1)).mean())