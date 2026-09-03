import torch
import numpy as np

# ============================================================
# MODEL 27 - DEEP LEAKAGE CHECK
# ============================================================
# PURPOSE:
#   Investigate the suspicious relationship:
#       X[4]  <->  Y[:,:,0]
#       U[0]  <->  Y[:,:,0]
#
# NO TRAINING
# NO MODEL
# READ-ONLY AUDIT
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("MODEL 27 - DEEP LEAKAGE CHECK")
print("=" * 70)

# ============================================================
# 1. LOAD SAME DATA
# ============================================================

data = torch.load(
    ".\\generated_data_ramp_10h\\combo_0000.pt",
    map_location="cpu",
    weights_only=False
)

print("\nDATA KEYS:")
print(data.keys())

X = data["X"].double()
Y = data["Y"].double()
U = data["U"].double()

target = Y[:, :, 0]

print("\nSHAPES")
print("X:", X.shape)
print("Y:", Y.shape)
print("U:", U.shape)
print("Target:", target.shape)

# ============================================================
# 2. EXACT NUMERICAL RELATIONSHIP X[4] vs TARGET
# ============================================================

print("\n" + "=" * 70)
print("TEST 1 - X[4] VS TARGET")
print("=" * 70)

x4 = X[:, :, 4]

difference = x4 - target

print("\nX[4] statistics:")
print("min :", x4.min().item())
print("max :", x4.max().item())
print("mean:", x4.mean().item())
print("std :", x4.std().item())

print("\nTarget statistics:")
print("min :", target.min().item())
print("max :", target.max().item())
print("mean:", target.mean().item())
print("std :", target.std().item())

print("\nDifference X[4] - Target:")
print("min :", difference.min().item())
print("max :", difference.max().item())
print("mean:", difference.mean().item())
print("std :", difference.std().item())

print(
    "\nMean absolute difference:",
    torch.abs(difference).mean().item()
)

print(
    "Maximum absolute difference:",
    torch.abs(difference).max().item()
)

print(
    "All values exactly equal:",
    torch.equal(x4, target)
)

print(
    "All values approximately equal:",
    torch.allclose(x4, target, rtol=1e-6, atol=1e-8)
)

# ============================================================
# 3. CHECK WHETHER X[4] IS A SIMPLE TRANSFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TEST 2 - POSSIBLE TRANSFORMATION X[4] -> TARGET")
print("=" * 70)

x = x4.reshape(-1)
y = target.reshape(-1)

x_mean = x.mean()
y_mean = y.mean()

x_centered = x - x_mean
y_centered = y - y_mean

slope = (
    (x_centered * y_centered).sum()
    / (x_centered ** 2).sum()
)

intercept = y_mean - slope * x_mean

predicted_y = slope * x + intercept

residual = y - predicted_y

ss_res = (residual ** 2).sum()
ss_tot = ((y - y_mean) ** 2).sum()

r2 = 1.0 - ss_res / ss_tot

print("\nLinear fit:")
print("Target ≈ slope * X[4] + intercept")
print("slope    :", slope.item())
print("intercept:", intercept.item())
print("R^2      :", r2.item())

print("\nResidual statistics:")
print("mean abs residual:", torch.abs(residual).mean().item())
print("max abs residual :", torch.abs(residual).max().item())
print("std residual     :", residual.std().item())

# ============================================================
# 4. CHECK TIME ALIGNMENT OF X[4] AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("TEST 3 - TEMPORAL ALIGNMENT X[4] VS TARGET")
print("=" * 70)

for shift in range(-5, 6):

    if shift > 0:
        a = x4[:, shift:]
        b = target[:, :-shift]

    elif shift < 0:
        a = x4[:, :shift]
        b = target[:, -shift:]

    else:
        a = x4
        b = target

    a = a.reshape(-1)
    b = b.reshape(-1)

    ac = a - a.mean()
    bc = b - b.mean()

    denominator = (
        torch.sqrt((ac ** 2).sum())
        * torch.sqrt((bc ** 2).sum())
    )

    if denominator > 0:
        corr = (
            (ac * bc).sum() / denominator
        ).item()
    else:
        corr = 0.0

    print(
        f"shift {shift:+d}: correlation = {corr:.9f}"
    )

# ============================================================
# 5. U[0] VS TARGET
# ============================================================

print("\n" + "=" * 70)
print("TEST 4 - U[0] VS TARGET")
print("=" * 70)

u0 = U[:, :, 0]

difference_u = u0 - target

print("\nU[0] statistics:")
print("min :", u0.min().item())
print("max :", u0.max().item())
print("mean:", u0.mean().item())
print("std :", u0.std().item())

print("\nDifference U[0] - Target:")
print("mean abs difference:",
      torch.abs(difference_u).mean().item())

print("max abs difference:",
      torch.abs(difference_u).max().item())

# correlation
u = u0.reshape(-1)
y = target.reshape(-1)

uc = u - u.mean()
yc = y - y.mean()

denominator = (
    torch.sqrt((uc ** 2).sum())
    * torch.sqrt((yc ** 2).sum())
)

corr_u0 = (
    (uc * yc).sum() / denominator
).item()

print("\nCorrelation U[0] vs Target:")
print(corr_u0)

# ============================================================
# 6. FIRST TRAJECTORY SIDE-BY-SIDE
# ============================================================

print("\n" + "=" * 70)
print("TEST 5 - FIRST TRAJECTORY")
print("=" * 70)

print("\nFirst 20 timesteps:")
print("\n t        X[4]          Target        U[0]")

for t in range(min(20, X.shape[1])):

    print(
        f"{t:2d}   "
        f"{x4[0,t].item():12.6f}   "
        f"{target[0,t].item():12.6f}   "
        f"{u0[0,t].item():12.6f}"
    )

# ============================================================
# 7. CHECK WHETHER X[4] IS JUST ANOTHER Y VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("TEST 6 - X[4] VS ALL Y VARIABLES")
print("=" * 70)

for j in range(Y.shape[2]):

    a = X[:, :, 4].reshape(-1)
    b = Y[:, :, j].reshape(-1)

    ac = a - a.mean()
    bc = b - b.mean()

    denominator = (
        torch.sqrt((ac ** 2).sum())
        * torch.sqrt((bc ** 2).sum())
    )

    if denominator > 0:

        corr = (
            (ac * bc).sum() / denominator
        ).item()

    else:
        corr = 0.0

    print(
        f"X[4] vs Y[{j}]: correlation = {corr:.9f}"
    )

# ============================================================
# 8. CHECK WHETHER U[0] IS JUST ANOTHER Y VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("TEST 7 - U[0] VS ALL Y VARIABLES")
print("=" * 70)

for j in range(Y.shape[2]):

    a = U[:, :, 0].reshape(-1)
    b = Y[:, :, j].reshape(-1)

    ac = a - a.mean()
    bc = b - b.mean()

    denominator = (
        torch.sqrt((ac ** 2).sum())
        * torch.sqrt((bc ** 2).sum())
    )

    if denominator > 0:

        corr = (
            (ac * bc).sum() / denominator
        ).item()

    else:
        corr = 0.0

    print(
        f"U[0] vs Y[{j}]: correlation = {corr:.9f}"
    )

# ============================================================
# 9. FINAL AUTOMATIC INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

exact_match = torch.equal(x4, target)

approx_match = torch.allclose(
    x4,
    target,
    rtol=1e-6,
    atol=1e-8
)

if exact_match:

    print(
        "\nCRITICAL:"
        "\nX[4] IS EXACTLY THE TARGET."
        "\nThis is direct target leakage."
        "\nDO NOT TRAIN MODEL 27."
    )

elif approx_match:

    print(
        "\nCRITICAL:"
        "\nX[4] IS NUMERICALLY EQUIVALENT TO THE TARGET."
        "\nThis is target leakage."
        "\nDO NOT TRAIN MODEL 27."
    )

elif r2 > 0.999999:

    print(
        "\nCRITICAL:"
        "\nX[4] almost perfectly reconstructs the target."
        "\nThis is highly suspicious and must be investigated."
        "\nDO NOT TRAIN MODEL 27 yet."
    )

elif r2 > 0.99:

    print(
        "\nWARNING:"
        "\nX[4] explains more than 99% of target variance."
        "\nThis requires investigation before training."
    )

else:

    print(
        "\nX[4] does not appear to be a direct copy"
        "\nof the target."
    )

print("\nU[0] correlation with target:", corr_u0)

if corr_u0 > 0.99:

    print(
        "\nWARNING:"
        "\nU[0] is extremely strongly correlated with target."
        "\nCheck whether U[0] is a legitimate known input"
        "\nor a variable derived from the future target."
    )

print("\n" + "=" * 70)
print("DEEP LEAKAGE CHECK COMPLETED")
print("=" * 70)

print("\nNO MODEL TRAINING WAS PERFORMED.")
print("DO NOT CHANGE THE DATASET YET.")
print("SEND ME THE COMPLETE OUTPUT OF THIS SCRIPT.")