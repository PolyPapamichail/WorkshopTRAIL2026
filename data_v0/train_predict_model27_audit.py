import torch
import numpy as np

# ============================================================
# MODEL 27 - DATA / X-Y-U AUDIT
# NO TRAINING
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("MODEL 27 - DATA AUDIT")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
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

print("\nRAW SHAPES")
print("X:", X.shape)
print("Y:", Y.shape)
print("U:", U.shape)

# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA INFORMATION")
print("=" * 70)

print("\nX")
print("shape:", X.shape)
print("min:", X.min().item())
print("max:", X.max().item())
print("mean:", X.mean().item())
print("std:", X.std().item())

print("\nY")
print("shape:", Y.shape)
print("min:", Y.min().item())
print("max:", Y.max().item())
print("mean:", Y.mean().item())
print("std:", Y.std().item())

print("\nU")
print("shape:", U.shape)
print("min:", U.min().item())
print("max:", U.max().item())
print("mean:", U.mean().item())
print("std:", U.std().item())

# ============================================================
# 3. TARGET
# ============================================================

target = Y[:, :, 0]

print("\n" + "=" * 70)
print("TARGET = Y[:, :, 0]")
print("=" * 70)

print("Target shape:", target.shape)
print("Target min:", target.min().item())
print("Target max:", target.max().item())
print("Target mean:", target.mean().item())
print("Target std:", target.std().item())

# ============================================================
# 4. PER-FEATURE X STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("X FEATURE STATISTICS")
print("=" * 70)

for i in range(X.shape[2]):

    feature = X[:, :, i]

    print(
        f"X[{i}] "
        f"min={feature.min().item():.6f} "
        f"max={feature.max().item():.6f} "
        f"mean={feature.mean().item():.6f} "
        f"std={feature.std().item():.6f}"
    )

# ============================================================
# 5. PER-FEATURE U STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("U FEATURE STATISTICS")
print("=" * 70)

for i in range(U.shape[2]):

    feature = U[:, :, i]

    print(
        f"U[{i}] "
        f"min={feature.min().item():.6f} "
        f"max={feature.max().item():.6f} "
        f"mean={feature.mean().item():.6f} "
        f"std={feature.std().item():.6f}"
    )

# ============================================================
# 6. PER-VARIABLE Y STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("Y VARIABLE STATISTICS")
print("=" * 70)

for i in range(Y.shape[2]):

    variable = Y[:, :, i]

    print(
        f"Y[{i}] "
        f"min={variable.min().item():.6f} "
        f"max={variable.max().item():.6f} "
        f"mean={variable.mean().item():.6f} "
        f"std={variable.std().item():.6f}"
    )

# ============================================================
# 7. CHECK X FEATURES AGAINST TARGET
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION: X FEATURES vs TARGET")
print("=" * 70)

target_flat = target.reshape(-1)

for i in range(X.shape[2]):

    x_flat = X[:, :, i].reshape(-1)

    x_centered = x_flat - x_flat.mean()
    y_centered = target_flat - target_flat.mean()

    denominator = (
        torch.sqrt((x_centered ** 2).sum()) *
        torch.sqrt((y_centered ** 2).sum())
    )

    if denominator > 0:
        correlation = (
            (x_centered * y_centered).sum() /
            denominator
        ).item()
    else:
        correlation = 0.0

    print(
        f"X[{i}] vs Target: "
        f"correlation = {correlation:.6f}"
    )

# ============================================================
# 8. CHECK U FEATURES AGAINST TARGET
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION: U FEATURES vs TARGET")
print("=" * 70)

for i in range(U.shape[2]):

    u_flat = U[:, :, i].reshape(-1)

    u_centered = u_flat - u_flat.mean()
    y_centered = target_flat - target_flat.mean()

    denominator = (
        torch.sqrt((u_centered ** 2).sum()) *
        torch.sqrt((y_centered ** 2).sum())
    )

    if denominator > 0:
        correlation = (
            (u_centered * y_centered).sum() /
            denominator
        ).item()
    else:
        correlation = 0.0

    print(
        f"U[{i}] vs Target: "
        f"correlation = {correlation:.6f}"
    )

# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

n_total = X.shape[0]
n_train = int(0.8 * n_total)

X_train = X[:n_train]
X_test = X[n_train:]

Y_train = Y[:n_train]
Y_test = Y[n_train:]

U_train = U[:n_train]
U_test = U[n_train:]

target_train = target[:n_train]
target_test = target[n_train:]

print("\n" + "=" * 70)
print("TRAIN / TEST DISTRIBUTION")
print("=" * 70)

print("\nTRAIN")
print("X mean:", X_train.mean().item())
print("X std :", X_train.std().item())
print("Y mean:", Y_train.mean().item())
print("Y std :", Y_train.std().item())
print("U mean:", U_train.mean().item())
print("U std :", U_train.std().item())
print("Target mean:", target_train.mean().item())
print("Target std :", target_train.std().item())

print("\nTEST")
print("X mean:", X_test.mean().item())
print("X std :", X_test.std().item())
print("Y mean:", Y_test.mean().item())
print("Y std :", Y_test.std().item())
print("U mean:", U_test.mean().item())
print("U std :", U_test.std().item())
print("Target mean:", target_test.mean().item())
print("Target std :", target_test.std().item())

# ============================================================
# 10. FIRST TRAJECTORY
# ============================================================

print("\n" + "=" * 70)
print("FIRST TRAIN TRAJECTORY")
print("=" * 70)

print("\nTarget first trajectory:")
print(target_train[0])

print("\nX first trajectory:")
print(X_train[0])

print("\nU first trajectory:")
print(U_train[0])

print("\nY first trajectory - first variable:")
print(Y_train[0, :, 0])

# ============================================================
# 11. CHECK WHETHER TARGET APPEARS IN X
# ============================================================

print("\n" + "=" * 70)
print("DIRECT TARGET MATCH CHECK")
print("=" * 70)

for i in range(X.shape[2]):

    difference = torch.abs(
        X[:, :, i] - target
    )

    max_difference = difference.max().item()
    mean_difference = difference.mean().item()

    print(
        f"X[{i}] vs Target | "
        f"mean abs diff = {mean_difference:.10f} | "
        f"max abs diff = {max_difference:.10f}"
    )

# ============================================================
# 12. TARGET TEMPORAL DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("TARGET TEMPORAL STRUCTURE")
print("=" * 70)

target_difference = target[:, 1:] - target[:, :-1]

print(
    "Target delta min:",
    target_difference.min().item()
)

print(
    "Target delta max:",
    target_difference.max().item()
)

print(
    "Target delta mean:",
    target_difference.mean().item()
)

print(
    "Target delta std:",
    target_difference.std().item()
)

# ============================================================
# 13. UNIQUE / CONSTANT CHECK
# ============================================================

print("\n" + "=" * 70)
print("LOW-VARIANCE FEATURE CHECK")
print("=" * 70)

for i in range(X.shape[2]):

    std = X[:, :, i].std().item()

    if std < 1e-8:
        print(f"WARNING: X[{i}] is effectively CONSTANT")

for i in range(U.shape[2]):

    std = U[:, :, i].std().item()

    if std < 1e-8:
        print(f"WARNING: U[{i}] is effectively CONSTANT")

for i in range(Y.shape[2]):

    std = Y[:, :, i].std().item()

    if std < 1e-8:
        print(f"WARNING: Y[{i}] is effectively CONSTANT")

# ============================================================
# 14. WINDOW ALIGNMENT TEST
# ============================================================

print("\n" + "=" * 70)
print("WINDOW ALIGNMENT TEST")
print("=" * 70)

sequence_length = 5
prediction_horizon = 5

t = 0

x_window = torch.cat(
    [
        X_train[0, t:t + sequence_length, :],
        U_train[0, t:t + sequence_length, :]
    ],
    dim=1
)

y_window = target_train[
    0,
    t + 1:t + prediction_horizon + 1
]

print("\nInput timesteps:")
print(list(range(t, t + sequence_length)))

print("\nTarget timesteps:")
print(list(range(
    t + 1,
    t + prediction_horizon + 1
)))

print("\nX window shape:", x_window.shape)
print("Y window shape:", y_window.shape)

print("\nTarget values used:")
print(y_window)

# ============================================================
# 15. FINAL
# ============================================================

print("\n" + "=" * 70)
print("MODEL 27 AUDIT COMPLETED")
print("=" * 70)
print("NO MODEL TRAINING WAS PERFORMED.")
print("SEND ME THE COMPLETE OUTPUT.")