import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
import os

# ============================================================
# MODEL 23
# Improved MLP / Residual architecture
# Based on the verified model 22 data pipeline
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================

data = torch.load(
    ".\\generated_data_ramp_10h\\combo_0000.pt",
    map_location="cpu",
    weights_only=False
)

print(data.keys())

X = data["X"]
Y = data["Y"]
U = data["U"]

print("X:", X.shape)
print("Y:", Y.shape)
print("U:", U.shape)

# ============================================================
# 2. TARGET
# ============================================================

# Target is the first variable from Y
target = Y[:, :, 0]

print("Target shape:", target.shape)

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

n_total = X.shape[0]
n_train = int(0.8 * n_total)

X_train_raw = X[:n_train]
X_test_raw = X[n_train:]

Y_train_raw = Y[:n_train]
Y_test_raw = Y[n_train:]

U_train_raw = U[:n_train]
U_test_raw = U[n_train:]

target_train_raw = target[:n_train]
target_test_raw = target[n_train:]

print("Total trajectories:", n_total)
print("Training trajectories:", target_train_raw.shape)
print("Testing trajectories:", target_test_raw.shape)

# ============================================================
# 4. BUILD INPUT / TARGET WINDOWS
# ============================================================

# We use 5-step prediction horizon.
#
# For every time t:
#
# Input:
#   5 consecutive time points
#   X + U information
#
# Target:
#   target at t+1 ... t+5
#
# This creates:
#
# 120 trajectories
# x 92 valid starting positions
# = 11040 samples
#
# ============================================================

sequence_length = 5
prediction_horizon = 5

# ------------------------------------------------------------
# X has 8 variables
# U has 4 variables
#
# Total features per timestep = 12
# ------------------------------------------------------------

num_X_features = X_train_raw.shape[2]
num_U_features = U_train_raw.shape[2]

num_features = num_X_features + num_U_features

print("X features:", num_X_features)
print("U features:", num_U_features)
print("Total features per timestep:", num_features)

# ------------------------------------------------------------
# Number of valid samples
# 101 - 5 + 1 - 5 + 1 = 93
#
# However, to stay consistent with the verified model 22
# pipeline we use 92 samples.
# ------------------------------------------------------------

n_samples_per_trajectory = 92

X_train_sequences = []
y_train_sequences = []

X_test_sequences = []
y_test_sequences = []

# ============================================================
# TRAIN WINDOWS
# ============================================================

for traj in range(n_train):

    for t in range(n_samples_per_trajectory):

        # 5 consecutive timesteps
        x_window = torch.cat(
            [
                X_train_raw[traj, t:t + sequence_length, :],
                U_train_raw[traj, t:t + sequence_length, :]
            ],
            dim=1
        )

        # Future target:
        # t+1 ... t+5
        y_window = target_train_raw[
            traj,
            t + 1:t + prediction_horizon + 1
        ]

        X_train_sequences.append(x_window)
        y_train_sequences.append(y_window)

# ============================================================
# TEST WINDOWS
# ============================================================

for traj in range(X_test_raw.shape[0]):

    for t in range(n_samples_per_trajectory):

        x_window = torch.cat(
            [
                X_test_raw[traj, t:t + sequence_length, :],
                U_test_raw[traj, t:t + sequence_length, :]
            ],
            dim=1
        )

        y_window = target_test_raw[
            traj,
            t + 1:t + prediction_horizon + 1
        ]

        X_test_sequences.append(x_window)
        y_test_sequences.append(y_window)

# ============================================================
# STACK
# ============================================================

X_train = torch.stack(X_train_sequences)
y_train = torch.stack(y_train_sequences)

X_test = torch.stack(X_test_sequences)
y_test = torch.stack(y_test_sequences)

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

print("Testing features:", X_test.shape)
print("Testing target:", y_test.shape)

# ============================================================
# 5. CHECK DATASET
# ============================================================

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

assert X_train.shape[1] == sequence_length
assert X_train.shape[2] == num_features

assert y_train.shape[1] == prediction_horizon

print("Dataset sizes are compatible.")

# ============================================================
# 6. FLATTEN SEQUENCES
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test = X_test.reshape(
    X_test.shape[0],
    -1
)

print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)

print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)

# ============================================================
# FINAL SAFETY CHECK
# ============================================================

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

# Expected:
#
# X_train = [11040, 60]
# y_train = [11040, 5]
#
# X_test  = [2760, 60]
# y_test  = [2760, 5]
#
# ============================================================

# ============================================================
# 7. NORMALIZATION
# ============================================================

X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True)

X_std[X_std < 1e-8] = 1.0

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# ------------------------------------------------------------
# Normalize target using training targets only
# ------------------------------------------------------------

y_mean = y_train.mean(dim=0, keepdim=True)
y_std = y_train.std(dim=0, keepdim=True)

y_std[y_std < 1e-8] = 1.0

y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

print("Normalization completed.")
print(
    "Normalized X_train mean:",
    X_train_norm.mean().item()
)
print(
    "Normalized X_train std:",
    X_train_norm.std().item()
)

# ============================================================
# 8. DATASET / DATALOADER
# ============================================================

dataset = TensorDataset(
    X_train_norm.float(),
    y_train_norm.float()
)

batch_size = 256

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

print("Training samples:", len(dataset))
print("Batch size:", batch_size)

# ============================================================
# 9. RESIDUAL MLP
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

        x = self.block(x)

        x = x + residual

        return self.activation(x)


class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_size,
        output_size
    ):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU()
        )

        self.residual1 = ResidualBlock(
            256,
            dropout=0.05
        )

        self.residual2 = ResidualBlock(
            256,
            dropout=0.05
        )

        self.residual3 = ResidualBlock(
            256,
            dropout=0.05
        )

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

        x = self.output_layer(x)

        return x


# ============================================================
# 10. MODEL
# ============================================================

input_size = X_train_norm.shape[1]
output_size = y_train_norm.shape[1]

model = ResidualMLP(
    input_size=input_size,
    output_size=output_size
)

print(model)

# ============================================================
# 11. LOSS / OPTIMIZER
# ============================================================

loss_function = nn.MSELoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=15
)

# ============================================================
# 12. TRAINING
# ============================================================

epochs = 200

best_loss = float("inf")
best_state = None

for epoch in range(epochs):

    model.train()

    epoch_loss = 0.0

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = loss_function(
            prediction,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += (
            loss.item() *
            batch_X.size(0)
        )

    epoch_loss /= len(dataset)

    scheduler.step(epoch_loss)

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

    if (epoch + 1) % 10 == 0:

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {epoch_loss:.6f} "
            f"- LR: {current_lr:.6f}"
        )

# ============================================================
# 13. RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(best_state)

print("Best training loss:", best_loss)

# ============================================================
# 14. TEST
# ============================================================

model.eval()

with torch.no_grad():

    prediction_norm = model(
        X_test_norm.float()
    )

# ============================================================
# 15. DENORMALIZE PREDICTIONS
# ============================================================

prediction = (
    prediction_norm *
    y_std +
    y_mean
)

prediction = prediction.double()

real_values = y_test.double()

print(
    "Test prediction shape:",
    prediction.shape
)

print(
    "Real test values shape:",
    real_values.shape
)

# ============================================================
# 16. PRINT FIRST 10
# ============================================================

print("First 10 real values:")
print(real_values[:10])

print("First 10 predicted values:")
print(prediction[:10])

# ============================================================
# 17. OVERALL METRICS
# ============================================================

absolute_error = torch.abs(
    prediction - real_values
)

squared_error = (
    prediction - real_values
) ** 2

mae = absolute_error.mean().item()

rmse = torch.sqrt(
    squared_error.mean()
).item()

print()
print("=================================")
print("OVERALL RESULTS")
print("=================================")

print("MAE:", mae)
print("RMSE:", rmse)

# ============================================================
# 18. RESULTS BY HORIZON
# ============================================================

print()
print("=================================")
print("RESULTS BY HORIZON")
print("=================================")

for h in range(prediction_horizon):

    horizon_error = (
        prediction[:, h] -
        real_values[:, h]
    )

    horizon_mae = torch.abs(
        horizon_error
    ).mean().item()

    horizon_rmse = torch.sqrt(
        (horizon_error ** 2).mean()
    ).item()

    print(
        f"Horizon {h + 1}: "
        f"MAE = {horizon_mae:.6f}, "
        f"RMSE = {horizon_rmse:.6f}"
    )

# ============================================================
# 19. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_size": input_size,
        "output_size": output_size,
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon
    },
    "model23_best.pt"
)

print()
print("Model saved as model23_best.pt")
print("MODEL 23 COMPLETED.")