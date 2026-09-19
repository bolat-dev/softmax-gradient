# Softmax + Cross-Entropy Gradient

Hand-derived gradient of the softmax cross-entropy loss, implemented from scratch in NumPy.

$$\frac{\partial L}{\partial z_j} = \hat{y}_j - y_j, \qquad
\frac{\partial L}{\partial W_{jm}} = (\hat{y}_j - y_j)\,x_m$$

- [Full derivation (PDF)](softmax_regression.pdf)
- [NumPy implementation](softmax_regression.py)

I derived this by hand while learning index notation and the chain rule, then checked it against a from-scratch training loop.
