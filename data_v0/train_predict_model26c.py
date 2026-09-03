import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
INPUT_WINDOW = 5
HORIZON = 5

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.001

DATA_PATH = r".\generated_data_ramp_10h\combo_0000.pt"
SAVE_PATH = r".\model26c_best.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# LOAD DATA
# ============================================================

data = torch.load(DATA_PATH, map_location="cpu")

X = data["X"].double()
Y = data["Y"].double()
U = data["U"].double()

target = Y[:, :, 0]

print("X:", X.shape)
print("Y:", Y.shape)
print("U:", U.shape)
print("Target shape:", target.shape)


# ============================================================
# TRAJECTORY SPLIT
# Same split as Model26 / Model26b
# ============================================================

num_trajectories = X.shape[0]

train_pool_size = int(0.8 * num_trajectories)

train_pool_indices = list(range(train_pool_size))
test_indices = list(range(train_pool_size, num_trajectories))

generator = torch.Generator().manual_seed(SEED)
perm = torch.randperm(train_pool_size, generator=generator).tolist()

val_size = int(0.10 * train_pool_size)

validation_indices = [train_pool_indices[i] for i in perm[:val_size]]
train_indices = [train_pool_indices[i] for i in perm[val_size:]]

print("\n=================================")
print("TRAJECTORY SPLIT")
print("=================================")
print("Total trajectories:", num_trajectories)
print("Training trajectories:", len(train_indices))
print("Validation trajectories:", len(validation_indices))
print("Testing trajectories:", len(test_indices))


# ============================================================
# BUILD DELTA-Y DATASET
#
# Input:
#   X(t-5 ... t-1)
#   U(t-5 ... t-1)
#   Y(t-1)
#
# Target:
#   Delta Y =
#   Y(t ... t+4) - Y(t-1)
#
# Final prediction:
#   Y_hat = Y(t-1) + Delta_Y_hat
# ============================================================

def build_dataset(X_data, U_data, Y_data):

    features = []
    delta_targets = []
    last_y_values = []

    n_traj = X_data.shape[0]
    n_time = X_data.shape[1]

    for traj in range(n_traj):

        for t in range(INPUT_WINDOW, n_time - HORIZON + 1):

            # Previous 5 X values
            x_window = X_data[traj, t - INPUT_WINDOW:t, :]

            # Previous 5 U values
            u_window = U_data[traj, t - INPUT_WINDOW:t, :]

            # Last observed target Y(t-1)
            last_y = Y_data[traj, t - 1]

            # Repeat Y(t-1) across the input window
            last_y_feature = torch.full(
                (INPUT_WINDOW, 1),
                last_y.item(),
                dtype=x_window.dtype
            )

            # 8 X + 4 U + 1 previous Y = 13 features
            combined = torch.cat(
                [x_window, u_window, last_y_feature],
                dim=-1
            )

            # Future true Y
            y_future = Y_data[traj, t:t + HORIZON]

            # Delta relative to last observed Y
            delta_y = y_future - last_y

            features.append(combined)
            delta_targets.append(delta_y)
            last_y_values.append(last_y)

    return (
        torch.stack(features),
        torch.stack(delta_targets),
        torch.stack(last_y_values)
    )


# ============================================================
# CREATE SPLIT DATA
# ============================================================

X_train_raw = X[train_indices]
U_train_raw = U[train_indices]
Y_train_raw = target[train_indices]

X_val_raw = X[validation_indices]
U_val_raw = U[validation_indices]
Y_val_raw = target[validation_indices]

X_test_raw = X[test_indices]
U_test_raw = U[test_indices]
Y_test_raw = target[test_indices]


X_train_seq, y_train_delta, last_y_train = build_dataset(
    X_train_raw,
    U_train_raw,
    Y_train_raw
)

X_val_seq, y_val_delta, last_y_val = build_dataset(
    X_val_raw,
    U_val_raw,
    Y_val_raw
)

X_test_seq, y_test_delta, last_y_test = build_dataset(
    X_test_raw,
    U_test_raw,
    Y_test_raw
)


print("\n=================================")
print("DELTA-Y WINDOW DATASETS")
print("=================================")

print("Training features:", X_train_seq.shape)
print("Training delta targets:", y_train_delta.shape)

print("Validation features:", X_val_seq.shape)
print("Validation delta targets:", y_val_delta.shape)

print("Testing features:", X_test_seq.shape)
print("Testing delta targets:", y_test_delta.shape)


# ============================================================
# FLATTEN
# 5 timesteps × 13 features = 65
# ============================================================

X_train = X_train_seq.reshape(X_train_seq.shape[0], -1).float()
X_val = X_val_seq.reshape(X_val_seq.shape[0], -1).float()
X_test = X_test_seq.reshape(X_test_seq.shape[0], -1).float()

y_train = y_train_delta.float()
y_val = y_val_delta.float()
y_test = y_test_delta.float()

last_y_train = last_y_train.float()
last_y_val = last_y_val.float()
last_y_test = last_y_test.float()

print("\nFinal X_train:", X_train.shape)
print("Final y_train:", y_train.shape)
print("Final X_val:", X_val.shape)
print("Final y_val:", y_val.shape)
print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)


# ============================================================
# NORMALIZATION
# TRAINING DATA ONLY
# ============================================================

X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True)

X_std[X_std < 1e-8] = 1.0

X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

print("\nNormalization completed.")
print("Normalized X_train mean:", X_train_norm.mean().item())
print("Normalized X_train std:", X_train_norm.std().item())


# ============================================================
# MODEL
# Same architecture as Model26 / Model26b
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, features):

        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(features, features)
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        return self.activation(
            x + self.block(x)
        )


class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=256,
        output_size=5
    ):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.residual1 = ResidualBlock(hidden_size)
        self.residual2 = ResidualBlock(hidden_size)
        self.residual3 = ResidualBlock(hidden_size)

        self.hidden = nn.Sequential(
            nn.Linear(hidden_size, 128),
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


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_norm,
    y_train
)

val_dataset = TensorDataset(
    X_val_norm,
    y_val
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)


# ============================================================
# MODEL SETUP
# ============================================================

model = ResidualMLP(
    input_size=65,
    hidden_size=256,
    output_size=HORIZON
).to(device)

print("\n")
print(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)

        loss = criterion(pred, yb)

        loss.backward()

        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)
        train_count += xb.size(0)

    train_loss = train_loss_sum / train_count


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            loss = criterion(pred, yb)

            val_loss_sum += loss.item() * xb.size(0)
            val_count += xb.size(0)

    val_loss = val_loss_sum / val_count


    # -------------------------
    # SAVE BEST
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }


    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f}"
        )


print("\nBest validation loss:", best_val_loss)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_state)
model.eval()


# ============================================================
# TEST
# ============================================================

with torch.no_grad():

    test_pred_delta = model(
        X_test_norm.to(device)
    ).cpu()

# Reconstruct absolute Y predictions
#
# Y_hat(t+h) =
# Y(t-1) + predicted_delta
#
pred_test = test_pred_delta + last_y_test.unsqueeze(1)

# True absolute Y
true_test = y_test + last_y_test.unsqueeze(1)


print("\nTest prediction shape:", pred_test.shape)
print("Real test values shape:", true_test.shape)


# ============================================================
# METRICS
# ============================================================

errors = pred_test - true_test

mae = torch.abs(errors).mean().item()

rmse = torch.sqrt(
    torch.mean(errors ** 2)
).item()


print("\n=================================")
print("MODEL 26C TEST RESULTS")
print("=================================")

print(f"Overall MAE : {mae:.6f}")
print(f"Overall RMSE: {rmse:.6f}")


print("\nHorizon metrics:")

for h in range(HORIZON):

    h_mae = torch.abs(
        errors[:, h]
    ).mean().item()

    h_rmse = torch.sqrt(
        torch.mean(errors[:, h] ** 2)
    ).item()

    print(
        f"H{h + 1}: "
        f"MAE = {h_mae:.6f}, "
        f"RMSE = {h_rmse:.6f}"
    )


# ============================================================
# PERSISTENCE BASELINE
#
# Predict Y(t+h) = Y(t-1)
# ============================================================

baseline_pred = last_y_test.unsqueeze(1).repeat(
    1,
    HORIZON
)

baseline_errors = baseline_pred - true_test

baseline_mae = torch.abs(
    baseline_errors
).mean().item()

baseline_rmse = torch.sqrt(
    torch.mean(baseline_errors ** 2)
).item()


print("\n=================================")
print("PERSISTENCE BASELINE")
print("=================================")

print(f"Baseline MAE : {baseline_mae:.6f}")
print(f"Baseline RMSE: {baseline_rmse:.6f}")


# ============================================================
# IMPROVEMENT
# ============================================================

mae_improvement = (
    (baseline_mae - mae)
    / baseline_mae
    * 100
)

rmse_improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
    * 100
)


print("\n=================================")
print("MODEL 26C vs BASELINE")
print("=================================")

print(
    f"MAE improvement : "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

torch.save(
    {
        "model_state_dict": best_state,
        "X_mean": X_mean,
        "X_std": X_std,
        "input_size": 65,
        "output_size": HORIZON,
        "input_window": INPUT_WINDOW,
        "horizon": HORIZON,
        "best_val_loss": best_val_loss,
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "train_trajectory_indices": train_indices,
        "validation_trajectory_indices": validation_indices,
        "test_trajectory_indices": test_indices,
    },
    SAVE_PATH
)

print("\nSaved:", SAVE_PATH)

print("\n=================================")
print("MODEL 26C COMPLETED")
print("=================================")