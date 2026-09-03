import torch
import numpy as np

# ============================================================
# MODEL 27 - LEAKAGE + TEMPORAL ALIGNMENT AUDIT
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("MODEL 27 - LEAKAGE + TEMPORAL ALIGNMENT AUDIT")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# SAME LOAD CODE AS PREVIOUS MODEL 27 AUDIT
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
# 2. BASIC CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("BASIC CONSISTENCY CHECK")
print("=" * 70)

assert X.ndim == 3
assert Y.ndim == 3
assert U.ndim == 3

assert X.shape[0] == Y.shape[0] == U.shape[0]
assert X.shape[1] == Y.shape[1] == U.shape[1]

print("Number of trajectories:", X.shape[0])
print("Number of timesteps:", X.shape[1])
print("X features:", X.shape[2])
print("Y variables:", Y.shape[2])
print("U features:", U.shape[2])

print("\nPASS: X, Y and U have compatible trajectory/time dimensions.")

# ============================================================
# 3. TARGET
# ============================================================

target = Y[:, :, 0]

print("\n" + "=" * 70)
print("TARGET")
print("=" * 70)

print("Target shape:", target.shape)
print("Target min:", target.min().item())
print("Target max:", target.max().item())
print("Target mean:", target.mean().item())
print("Target std:", target.std().item())

# ============================================================
# 4. DIRECT TARGET LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DIRECT TARGET LEAKAGE CHECK")
print("=" * 70)

print("\nChecking whether X contains target Y[:,:,0]...")

for i in range(X.shape[2]):

    difference = torch.abs(X[:, :, i] - target)

    mean_abs_diff = difference.mean().item()
    max_abs_diff = difference.max().item()

    exact_match = torch.allclose(
        X[:, :, i],
        target,
        atol=1e-10,
        rtol=1e-10
    )

    print(
        f"X[{i}] | "
        f"mean_abs_diff={mean_abs_diff:.10f} | "
        f"max_abs_diff={max_abs_diff:.10f} | "
        f"EXACT_MATCH={exact_match}"
    )

# ============================================================
# 5. CORRELATION X / TARGET
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
        torch.sqrt((x_centered ** 2).sum())
        * torch.sqrt((y_centered ** 2).sum())
    )

    if denominator > 0:

        correlation = (
            (x_centered * y_centered).sum()
            / denominator
        ).item()

    else:

        correlation = 0.0

    print(
        f"X[{i}] vs Target: "
        f"correlation = {correlation:.6f}"
    )

# ============================================================
# 6. CORRELATION U / TARGET
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION: U FEATURES vs TARGET")
print("=" * 70)

for i in range(U.shape[2]):

    u_flat = U[:, :, i].reshape(-1)

    u_centered = u_flat - u_flat.mean()
    y_centered = target_flat - target_flat.mean()

    denominator = (
        torch.sqrt((u_centered ** 2).sum())
        * torch.sqrt((y_centered ** 2).sum())
    )

    if denominator > 0:

        correlation = (
            (u_centered * y_centered).sum()
            / denominator
        ).item()

    else:

        correlation = 0.0

    print(
        f"U[{i}] vs Target: "
        f"correlation = {correlation:.6f}"
    )

# ============================================================
# 7. TEMPORAL ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL ALIGNMENT CHECK")
print("=" * 70)

sequence_length = 5
prediction_horizon = 5

t = 0

x_window = torch.cat(
    [
        X[0, t:t + sequence_length, :],
        U[0, t:t + sequence_length, :]
    ],
    dim=1
)

y_window = target[
    0,
    t + 1:t + prediction_horizon + 1
]

print("\nInput timesteps:")
print(list(range(t, t + sequence_length)))

print("\nTarget timesteps:")
print(
    list(
        range(
            t + 1,
            t + prediction_horizon + 1
        )
    )
)

print("\nX window shape:", x_window.shape)
print("Y window shape:", y_window.shape)

print("\nTarget values used:")
print(y_window)

# ============================================================
# 8. CHECK FOR OVERLAP BETWEEN INPUT AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("INPUT / TARGET TIME OVERLAP CHECK")
print("=" * 70)

input_times = set(
    range(
        t,
        t + sequence_length
    )
)

target_times = set(
    range(
        t + 1,
        t + prediction_horizon + 1
    )
)

overlap = input_times.intersection(target_times)

print("Input timesteps :", sorted(input_times))
print("Target timesteps:", sorted(target_times))
print("Overlap         :", sorted(overlap))

if len(overlap) > 0:

    print(
        "\nWARNING: Input and target windows overlap "
        "in time."
    )

else:

    print(
        "\nPASS: No temporal overlap between "
        "input and target windows."
    )

# ============================================================
# 9. CHECK TARGET SHIFT RELATIONSHIP
# ============================================================

print("\n" + "=" * 70)
print("TARGET SHIFT CHECK")
print("=" * 70)

for shift in [0, 1, 2, -1]:

    if shift >= 0:

        a = target[:, shift:]
        b = target[:, :target.shape[1] - shift]

    else:

        s = abs(shift)

        a = target[:, :target.shape[1] - s]
        b = target[:, s:]

    if a.numel() == 0:
        continue

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()

    denominator = (
        torch.sqrt((a_centered ** 2).sum())
        * torch.sqrt((b_centered ** 2).sum())
    )

    if denominator > 0:

        corr = (
            (a_centered * b_centered).sum()
            / denominator
        ).item()

    else:

        corr = 0.0

    print(
        f"Shift {shift:+d}: "
        f"correlation = {corr:.6f}"
    )

# ============================================================
# 10. TRAIN / TEST SPLIT CHECK
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT CHECK")
print("=" * 70)

n_total = X.shape[0]
n_train = int(0.8 * n_total)

train_indices = set(range(n_train))
test_indices = set(range(n_train, n_total))

overlap_indices = train_indices.intersection(test_indices)

print("Total trajectories:", n_total)
print("Train trajectories:", n_train)
print("Test trajectories :", n_total - n_train)
print("Index overlap     :", overlap_indices)

if len(overlap_indices) == 0:

    print(
        "\nPASS: Train and test trajectory indices "
        "do not overlap."
    )

else:

    print(
        "\nWARNING: Train/test trajectory overlap detected."
    )

# ============================================================
# 11. CHECK CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("LOW-VARIANCE / CONSTANT FEATURE CHECK")
print("=" * 70)

for i in range(X.shape[2]):

    std = X[:, :, i].std().item()

    if std < 1e-8:

        print(
            f"WARNING: X[{i}] is effectively CONSTANT "
            f"(std={std:.12e})"
        )

    else:

        print(
            f"X[{i}] std={std:.6f}"
        )

for i in range(U.shape[2]):

    std = U[:, :, i].std().item()

    if std < 1e-8:

        print(
            f"WARNING: U[{i}] is effectively CONSTANT "
            f"(std={std:.12e})"
        )

    else:

        print(
            f"U[{i}] std={std:.6f}"
        )

# ============================================================
# 12. TARGET TEMPORAL DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("TARGET TEMPORAL DIFFERENCES")
print("=" * 70)

target_difference = (
    target[:, 1:] - target[:, :-1]
)

print(
    "delta min :",
    target_difference.min().item()
)

print(
    "delta max :",
    target_difference.max().item()
)

print(
    "delta mean:",
    target_difference.mean().item()
)

print(
    "delta std :",
    target_difference.std().item()
)

# ============================================================
# 13. FIRST TRAJECTORY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FIRST TRAJECTORY SUMMARY")
print("=" * 70)

print("\nFirst target values:")
print(target[0, :20])

print("\nFirst U values:")
print(U[0, :10])

print("\nFirst X values:")
print(X[0, :10])

# ============================================================
# 14. FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("MODEL 27 LEAKAGE + ALIGNMENT AUDIT COMPLETED")
print("=" * 70)

print("NO MODEL TRAINING WAS PERFORMED.")

print("\nIMPORTANT:")
print(
    "The results above must be inspected before "
    "training Model 27."
)

print("=" * 70)