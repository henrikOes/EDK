import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m1, m2 = -1, 1
n_samples = 1000  # per class
variances = {1: 0.25, 2: 0.49, 3: 1.00}

rng = np.random.default_rng(seed=42)

os.makedirs("datasets", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, sigma2 in variances.items():
    sigma = np.sqrt(sigma2)

    class1 = rng.normal(loc=m1, scale=sigma, size=n_samples)
    class2 = rng.normal(loc=m2, scale=sigma, size=n_samples)

    # Stack: columns are [sample, label]; label 0 = class 1, label 1 = class 2
    data = np.column_stack([
        np.concatenate([class1, class2]),
        np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    ])

    filename = os.path.join("datasets", f"dataset_{i}_sigma2_{str(sigma2).replace('.', '')}.npy")
    np.save(filename, data)
    print(f"Dataset {i} (σ² = {sigma2}): saved as '{filename}' — shape: {data.shape}")

    # Plot
    ax = axes[i - 1]
    ax.hist(class1, bins=40, alpha=0.6, color="steelblue", label=f"Class 1 (m={m1})")
    ax.hist(class2, bins=40, alpha=0.6, color="tomato",    label=f"Class 2 (m={m2})")
    ax.set_title(f"Dataset {i}  —  σ² = {sigma2}")
    ax.set_xlabel("x")
    ax.set_ylabel("Count")
    ax.legend()

plt.suptitle("Gaussian class distributions (1 000 samples per class)", fontsize=13)
plt.tight_layout()
plt.savefig("datasets/distributions.png", dpi=150)
plt.show()
print("Plot saved to datasets/distributions.png")
