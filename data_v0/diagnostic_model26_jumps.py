import torch
import torch.nn as nn


# ============================================================
# 1. LOAD DATA + MODEL
# ============================================================

data = torch.load(
    r".\generated_data_ramp_10h\combo_0000.pt",
    map_location="cpu",
    weights_only=False
)

saved = torch.load(
    r".\model26_best.pt",
    map_location="cpu",
    weights_only=False
)

X = data["X"]
Y = data["Y"]
U = data["U"]

target = Y[:, :, 0].double()

INPUT_WINDOW = saved["input_window"]
HORIZON = saved["horizon"]


# ============================================================
# 2. SAME TRAIN / TEST SPLIT
# ============================================================

num_trajectories = X.shape[0]
train_pool_size = int(0.8 * num_trajectories)

X_train_pool = X[:train_pool_size]
U_train_pool = U[:train_pool_size]
Y_train_pool = target[:train_pool_size]

X_test_raw = X[train_pool_size:]
U_test_raw = U[train_pool_size:]
Y_test_raw = target[train_pool_size:]

num_train_pool = X_train_pool.shape[0]
num_val = max(1, int(num_train_pool * 0.10))

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    num_train_pool,
    generator=generator
)

val_indices = indices[:num_val]
train_indices = indices[num_val:]

X_train_raw = X_train_pool[train_indices]
U_train_raw = U_train_pool[train_indices]
Y_train_raw = Y_train_pool[train_indices]


# ============================================================
# 3. BUILD WINDOWS
# ============================================================

def build_dataset(X_data, U_data, Y_data):

    features = []
    targets = []
    last_observed_targets = []

    n_traj = X_data.shape[0]
    n_time = X_data.shape[1]

    for traj in range(n_traj):

        for t in range(
            INPUT_WINDOW,
            n_time - HORIZON + 1
        ):

            x_window = X_data[
                traj,
                t - INPUT_WINDOW:t,
                :
            ]

            u_window = U_data[
                traj,
                t - INPUT_WINDOW:t,
                :
            ]

            combined = torch.cat(
                [x_window, u_window],
                dim=-1
            )

            y_future = Y_data[
                traj,
                t:t + HORIZON
            ]

            # IMPORTANT:
            # actual last observed TARGET = Y(t-1)
            last_target = Y_data[
                traj,
                t - 1
            ]

            features.append(combined)
            targets.append(y_future)
            last_observed_targets.append(last_target)

    return (
        torch.stack(features),
        torch.stack(targets),
        torch.stack(last_observed_targets)
    )


X_train_seq, y_train, _ = build_dataset(
    X_train_raw,
    U_train_raw,
    Y_train_raw
)

X_test_seq, y_test, last_observed = build_dataset(
    X_test_raw,
    U_test_raw,
    Y_test_raw
)


# ============================================================
# 4. FLATTEN + NORMALIZE
# ============================================================

X_train = X_train_seq.reshape(
    X_train_seq.shape[0],
    -1
)

X_test = X_test_seq.reshape(
    X_test_seq.shape[0],
    -1
)

X_mean = X_train.mean(
    dim=0,
    keepdim=True
)

X_std = X_train.std(
    dim=0,
    keepdim=True
)

X_std[X_std < 1e-8] = 1.0

X_test_norm = (
    X_test - X_mean
) / X_std


# ============================================================
# 5. MODEL
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, features, dropout=0.05):

        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features, features)
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        residual = x
        out = self.block(x)
        out = out + residual

        return self.activation(out)


class ResidualMLP(nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU()
        )

        self.residual1 = ResidualBlock(256, 0.05)
        self.residual2 = ResidualBlock(256, 0.05)
        self.residual3 = ResidualBlock(256, 0.05)

        self.hidden = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(
            128,
            output_size
        )

    def forward(self, x):

        x = self.input_layer(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.hidden(x)

        return self.output_layer(x)


model = ResidualMLP(
    saved["input_size"],
    saved["output_size"]
)

model.load_state_dict(
    saved["model_state_dict"]
)

model.eval()


# ============================================================
# 6. PREDICTIONS
# ============================================================

with torch.no_grad():

    predictions = model(
        X_test_norm.float()
    ).double()

real_values = y_test.double()


# ============================================================
# 7. JUMP ERROR DIAGNOSTIC
# ============================================================

print()
print("=================================")
print("MODEL 26 CORRECT JUMP DIAGNOSTIC")
print("=================================")

for h in range(HORIZON):

    actual = real_values[:, h]

    # Change from the LAST OBSERVED TARGET Y(t-1)
    # to the actual future target Y(t+h)
    jump = (
        actual - last_observed
    ).abs()

    error = (
        predictions[:, h] - actual
    ).abs()

    stable = jump <= 0.10
    small = (jump > 0.10) & (jump <= 5.0)
    medium = (jump > 5.0) & (jump <= 20.0)
    large = jump > 20.0

    print()
    print(f"------------- H{h+1} -------------")

    for name, mask in [
        ("STABLE   |Δ| <= 0.1", stable),
        ("SMALL    0.1 < |Δ| <= 5", small),
        ("MEDIUM   5 < |Δ| <= 20", medium),
        ("LARGE    |Δ| > 20", large),
    ]:

        count = mask.sum().item()

        if count > 0:

            mae = error[mask].mean().item()

            print(
                f"{name:28s}: "
                f"N = {count:4d}, "
                f"MAE = {mae:.4f}"
            )

        else:

            print(
                f"{name:28s}: "
                f"N = 0"
            )


# ============================================================
# 8. OVERALL MAE
# ============================================================

print()
print("=================================")
print("OVERALL MAE BY HORIZON")
print("=================================")

for h in range(HORIZON):

    mae = (
        predictions[:, h] - real_values[:, h]
    ).abs().mean().item()

    print(
        f"H{h+1}: MAE = {mae:.6f}"
    )


print()
print("=================================")
print("CORRECT JUMP DIAGNOSTIC COMPLETED")
print("=================================")