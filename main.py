import torch
import pyro
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set random seed for reproducibility
pyro.set_rng_seed(101)


# Define the Generative Model
def generative_model(
    num_samples: int,
    x_cluster_locations: np.ndarray,
    x_cluster_scales: np.ndarray,
    alpha_for_deviance: float = 1.0,
    beta_for_deviance: float = 2.0,
    ps_for_positive_class: float = 0.2,
    ps_for_negative_class: float = 0.4,
):
    alpha_theta = torch.tensor(alpha_for_deviance)
    beta_theta = torch.tensor(beta_for_deviance)
    lognormal_scale = torch.from_numpy(x_cluster_scales)
    lognormal_locations = torch.from_numpy(x_cluster_locations)
    num_clusters = len(x_cluster_locations)
    dirichlet_base = 10.0 * torch.ones(num_clusters)  # Base concentrations
    dirichlet_k = torch.tensor(50.0)  # Scaling factor for theta dependency

    # Generative Process with Plate for Vectorization
    with pyro.plate("data", num_samples):
        # 1. Sample theta from Beta
        theta = pyro.sample("theta", dist.Beta(alpha_theta, beta_theta))

        # 2. Sample y from Binomial conditional on theta
        y = pyro.sample("y", dist.Binomial(probs=theta))

        # 3. Sample s (observation indicator for y) from Binomial conditional on y
        s = pyro.sample(
            "s",
            dist.Binomial(
                probs=(1 - y) * ps_for_negative_class + y * ps_for_positive_class
            ),
        )

        # 4. Calculate Dirichlet parameters based on theta
        concentration_params = torch.stack(
            [
                dirichlet_base[i]
                + dirichlet_k
                * (theta * (i % 2) + (1.0 - theta) * ((i + 1) % 2))
                / (i + 1)
                for i in range(num_clusters)
            ],
            dim=-1,
        )

        # 5. Sample pi (category probabilities) from Dirichlet
        pi = pyro.sample("pi", dist.Dirichlet(concentration_params))

        # 6. Sample c (category) from Categorical using pi
        c = pyro.sample("c", dist.Categorical(probs=pi))

        # 7. Sample x from LogNormal conditional on c
        selected_scale = lognormal_scale[c]
        selected_shape = lognormal_locations[c]
        x = pyro.sample("x", dist.LogNormal(loc=selected_shape, scale=selected_scale))

        # Mark y as observed only when s is 1
        pyro.deterministic("y_observed", torch.where(s == 1, y, torch.tensor(np.nan)))

    return theta, y, s, pi, c, x


# Generate Synthetic Data
num_synthetic_samples = 40000
num_clusters = 5
x_cluster_locations = np.linspace(0, 10, num_clusters)
x_cluster_scales = np.ones_like(x_cluster_locations)

# Generate data using the model
model = lambda num_samples: generative_model(
    num_samples,
    alpha_for_deviance=1.0,
    beta_for_deviance=4.0,
    x_cluster_locations=x_cluster_locations,
    x_cluster_scales=x_cluster_scales,
)

trace = pyro.poutine.trace(model).get_trace(num_samples=num_synthetic_samples)

# Extract the generated values
theta_samples = trace.nodes["theta"]["value"]
y_samples = trace.nodes["y"]["value"]
s_samples = trace.nodes["s"]["value"]
pi_samples = trace.nodes["pi"]["value"]
c_samples = trace.nodes["c"]["value"]
x_samples = trace.nodes["x"]["value"]
y_observed_samples = trace.nodes["y_observed"]["value"]

print(f"Generated {num_synthetic_samples} samples.")
print("Shapes:")
print(f"  theta: {theta_samples.shape}")
print(f"  y:     {y_samples.shape}")
print(f"  s:     {s_samples.shape}")
print(f"  pi:    {pi_samples.shape}")
print(f"  c:     {c_samples.shape}")
print(f"  x:     {x_samples.shape}")
print(f"  y_observed: {y_observed_samples.shape}")

# Inspect the Generated Data
df = pd.DataFrame(
    {
        "theta": theta_samples.numpy(),
        "theta_bin": np.round(theta_samples.numpy() / 0.10),
        "y": y_samples.numpy(),
        "s": s_samples.numpy(),
        "y_observed": y_observed_samples.numpy(),
        **{
            f"category_prob_{i}": pi_samples[:, i].numpy()
            for i in range(pi_samples.shape[1])
        },
        "category": c_samples.numpy(),
        "x": x_samples.numpy(),
    }
)

print("\nSampled Data Head:")
print(df.head())

print("\nCategory Counts:")
print(df["category"].value_counts())

print("\nObservation Counts for y:")
print(df["s"].value_counts())
print(f"Percentage of y observed: {df['s'].mean() * 100:.2f}%")

# Visualize the Results
plt.figure(figsize=(18, 12))

# Distribution of theta
plt.subplot(2, 2, 1)
sns.histplot(df["theta"], kde=True, bins=30)
plt.title("Distribution of theta")
plt.xlabel("theta")

# Calibration (using only observed y)
df_cal_observed = (
    df[df["s"] == 1].groupby("theta_bin").agg({"theta": "mean", "y": "mean"})
)
print("\nCalibration (Observed y only):")
print(df_cal_observed)
plt.subplot(2, 2, 2)
sns.scatterplot(x="theta", y="y", data=df_cal_observed, alpha=1)
plt.title("Calibration (Observed y only)")
plt.xlabel("Average theta")
plt.ylabel("Average observed y")

# Relationship between theta and category probabilities (pi)
plt.subplot(2, 2, 3)
for i in range(pi_samples.shape[1]):
    sns.scatterplot(
        x="theta", y=f"category_prob_{i}", data=df, alpha=0.2, label=f"Prob(c={i})"
    )
plt.title("Category Probabilities (pi) vs Theta")
plt.xlabel("Theta")
plt.ylabel("Probability")
plt.legend()

# Distribution of x
plt.subplot(2, 2, 4)
sns.kdeplot(x="x", data=df, hue="category", log_scale=(True, False), common_norm=True)
plt.title("Distribution of X")
plt.xlabel("X value (log scale)")

plt.tight_layout()
plt.show()

print("\nMean X value per category:")
print(df.groupby("category")["x"].mean())

print("\nMean X value per observed y and category:")
print(df[df["s"] == 1].groupby(["y", "category"])["x"].mean())

# %%
