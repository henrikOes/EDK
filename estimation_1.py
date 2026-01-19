import numpy as np
import matplotlib.pyplot as plt

# Parametre
A = 1.0
N = 1000
M = 500
eps = 1e-2
sigma2_s = 20
sigma_s = np.sqrt(sigma2_s)

sigma2_vals = np.logspace(-3, 0, 20)

var_mean = []
var_median = []

for sigma2 in sigma2_vals:
    sigma = np.sqrt(sigma2)

    est_mean = []
    est_median = []

    for _ in range(M):
        # Generer blandet støy
        w = np.zeros(N)
        for i in range(N):
            if np.random.rand() < eps:
                w[i] = abs(sigma_s * np.random.randn())   # shot noise
            else:
                w[i] = sigma * np.random.randn()          # vanlig Gaussisk støy

        x = A + w

        est_mean.append(np.mean(x))
        est_median.append(np.median(x))

    var_mean.append(np.var(est_mean, ddof=1))
    var_median.append(np.var(est_median, ddof=1))

# Plot
plt.figure()
plt.loglog(sigma2_vals, var_mean, 'o-', label='Sample mean')
plt.loglog(sigma2_vals, var_median, 's-', label='Median')
plt.xlabel(r'$\sigma^2$')
plt.ylabel('Variance')
plt.grid(True, which="both")
plt.legend()
plt.show()
