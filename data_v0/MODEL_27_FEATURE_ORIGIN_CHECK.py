import torch
import os

# ============================================================
# MODEL 27 - FEATURE ORIGIN CHECK
# ============================================================

print("=" * 70)
print("MODEL 27 - FEATURE ORIGIN CHECK")
print("=" * 70)

DATA_PATH = ".\\generated_data_ramp_10h\\combo_0000.pt"

data = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nDATA KEYS:")
print(data.keys())

X = data["X"].double()
Y = data["Y"].double()
U = data["U"].double()
param_combo = data["param_combo"]

target = Y[:, :, 0]

print("\nPARAM_COMBO:")
print(param_combo)

print("\nSHAPES:")
print("X:", X.shape)
print("Y:", Y.shape)
print("U:", U.shape)

# ============================================================
# 1. EXACT RELATION X[4] -> TARGET
# ============================================================

print("\n" + "=" * 70)
print("X[4] -> TARGET EXACT RELATION")
print("=" * 70)

ratio = target / X[:, :, 4]

print("Target / X[4]")
print("min :", ratio.min().item())
print("max :", ratio.max().item())
print("mean:", ratio.mean().item())
print("std :", ratio.std().item())

print("\nCheck Target - 3.3046 * X[4]")

residual = target - 3.3046 * X[:, :, 4]

print("mean abs residual:", residual.abs().mean().item())
print("max abs residual :", residual.abs().max().item())

# ============================================================
# 2. CHECK WHETHER X[4] IS A SIMPLE SCALED VERSION
# ============================================================

print("\n" + "=" * 70)
print("X[4] / TARGET SAMPLE")
print("=" * 70)

for traj in range(min(5, X.shape[0])):
    print(
        f"trajectory {traj}: "
        f"X[4,0]={X[traj,0,4].item():.10f} | "
        f"Target,0={target[traj,0].item():.10f} | "
        f"ratio={ratio[traj,0].item():.10f}"
    )

# ============================================================
# 3. CHECK ALL X FEATURES FOR SIMPLE LINEAR RELATION
# ============================================================

print("\n" + "=" * 70)
print("ALL X FEATURES VS TARGET")
print("=" * 70)

target_flat = target.reshape(-1)

for i in range(X.shape[2]):

    x = X[:, :, i].reshape(-1)

    x_mean = x.mean()
    y_mean = target_flat.mean()

    numerator = ((x - x_mean) * (target_flat - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()

    if denominator > 0:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        prediction = slope * x + intercept
        residual = target_flat - prediction

        ss_res = (residual ** 2).sum()
        ss_tot = ((target_flat - y_mean) ** 2).sum()

        r2 = 1.0 - ss_res / ss_tot

        print(
            f"X[{i}] | "
            f"slope={slope.item():.8f} | "
            f"intercept={intercept.item():.8f} | "
            f"R2={r2.item():.12f}"
        )

# ============================================================
# 4. ALL U FEATURES VS TARGET
# ============================================================

print("\n" + "=" * 70)
print("ALL U FEATURES VS TARGET")
print("=" * 70)

for i in range(U.shape[2]):

    u = U[:, :, i].reshape(-1)

    u_mean = u.mean()
    y_mean = target_flat.mean()

    numerator = ((u - u_mean) * (target_flat - y_mean)).sum()
    denominator = ((u - u_mean) ** 2).sum()

    if denominator > 0:
        slope = numerator / denominator
        intercept = y_mean - slope * u_mean

        prediction = slope * u + intercept
        residual = target_flat - prediction

        ss_res = (residual ** 2).sum()
        ss_tot = ((target_flat - y_mean) ** 2).sum()

        r2 = 1.0 - ss_res / ss_tot

        print(
            f"U[{i}] | "
            f"slope={slope.item():.8f} | "
            f"intercept={intercept.item():.8f} | "
            f"R2={r2.item():.12f}"
        )

# ============================================================
# 5. PARAM_COMBO STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("PARAM_COMBO STRUCTURE")
print("=" * 70)

print("type:", type(param_combo))

try:
    print("length:", len(param_combo))
except Exception:
    pass

print("value:")
print(param_combo)

# ============================================================
# 6. CHECK DATA DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("DATA DIRECTORY")
print("=" * 70)

base_dir = "."

for root, dirs, files in os.walk(base_dir):

    for file in files:

        if file.endswith(".py"):

            path = os.path.join(root, file)

            print(path)

print("\n" + "=" * 70)
print("FEATURE ORIGIN CHECK COMPLETED")
print("=" * 70)

print("NO DATA WAS MODIFIED.")
print("NO MODEL WAS TRAINED.")
print("SEND ME THE COMPLETE OUTPUT.")
