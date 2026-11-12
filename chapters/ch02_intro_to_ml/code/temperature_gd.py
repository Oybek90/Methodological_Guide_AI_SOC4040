# ci: run
"""Learn F = 32 + 1.8 t from five Celsius/Fahrenheit pairs with gradient descent."""
import numpy as np

from linreg import cost, gradient_descent

t = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
f = np.array([50.0, 59.0, 68.0, 77.0, 86.0])

# Stability limit: GD on a quadratic J converges iff alpha < 2 / lambda_max(H),
# where H = [[1, mean(x)], [mean(x), mean(x^2)]] is the Hessian of J.
H = np.array([[1.0, t.mean()], [t.mean(), (t ** 2).mean()]])
print("lambda_max =", round(float(np.linalg.eigvalsh(H).max()), 2), " -> alpha must be below", round(2 / float(np.linalg.eigvalsh(H).max()), 5))

# 1. Raw inputs: with t up to 30 the curvature is large, so alpha = 0.01 diverges.
bad = gradient_descent(t, f, alpha=0.01, steps=20)
print("raw t, alpha=0.01   :  J after 20 steps =", f"{bad[-1][2]:.3e}")

# 2. Raw inputs with a small enough alpha converge, but very slowly in theta_0.
slow = gradient_descent(t, f, alpha=0.001, steps=20000)
print("raw t, alpha=0.001  :  theta =", [round(v, 3) for v in slow[-1][:2]], f"J = {slow[-1][2]:.4f}")

# 3. Centre and scale the input first (z-score), then use a large alpha.
mu, sd = t.mean(), t.std()
z = (t - mu) / sd
fast = gradient_descent(z, f, alpha=0.5, steps=40)
b0, b1 = fast[-1][:2]
# convert back: F = b0 + b1 (t - mu)/sd   ->   theta1 = b1/sd,  theta0 = b0 - b1*mu/sd
theta1 = float(b1 / sd)
theta0 = float(b0 - b1 * mu / sd)
print("scaled t, alpha=0.5 :  theta =", [round(theta0, 3), round(theta1, 3)], f"J = {cost(theta0, theta1, t, f):.2e}")

# 4. Prediction for the unseen input of the lecture (t = 35 C).
print("prediction F(35 C) =", round(theta0 + theta1 * 35, 2), " (exact: 95.0)")
