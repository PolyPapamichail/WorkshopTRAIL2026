import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy

# ============================================================
# MODEL 24
# Strict generalization / leakage check
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

target = Y[:, :, 0]

print("Target shape:", target.shape)

# ============================================================
# 3. TRAJECTORY SPLIT
# ============================================================

n_total = X.shape[0]
n_train = int(0.8 * n_total)

X_train_raw = X[:n_train]
X_test_raw = X[n_train:]

U_train_raw = U[:n_train]
U_test_raw = U[n_train:]

target_train_raw = target[:n_train]
target_test_raw = target[n_train:]

print("Total trajectories:", n_total)
print("Training trajectories:", target_train_raw.shape)
print("Testing trajectories:", target_test_raw.shape)

# ============================================================
# 4. PARAMETERS
# ============================================================

sequence_length = 5
prediction_horizon = 5

num_X_features = X_train_raw.shape[2]
num_U_features = U_train_raw.shape[2]

num_features = num_X_features + num_U_features

print("X features:", num_X_features)
print("U features:", num_U_features)
print("Total features per timestep:", num_features)

n_samples_per_trajectory = 92

# ============================================================
# 5. BUILD WINDOWS
# ============================================================

X_train_sequences = []
y_train_sequences = []

X_test_sequences = []
y_test_sequences = []

# ============================================================
# TRAIN
# ============================================================

for traj in range(n_train):

    for t in range(n_samples_per_trajectory):

        x_window = torch.cat(
            [
                X_train_raw[
                    traj,
                    t:t + sequence_length,
                    :
                ],
                U_train_raw[
                    traj,
                    t:t + sequence_length,
                    :
                ]
            ],
            dim=1
        )

        y_window = target_train_raw[
            traj,
            t + 1:t + prediction_horizon + 1
        ]

        X_train_sequences.append(x_window)
        y_train_sequences.append(y_window)

# ============================================================
# TEST
# ============================================================

for traj in range(X_test_raw.shape[0]):

    for t in range(n_samples_per_trajectory):

        x_window = torch.cat(
            [
                X_test_raw[
                    traj,
                    t:t + sequence_length,
                    :
                ],
                U_test_raw[
                    traj,
                    t:t + sequence_length,
                    :
                ]
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
# 6. STACK
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
# 7. CHECK DATASET
# ============================================================

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

assert X_train.shape[1] == sequence_length
assert X_train.shape[2] == num_features

assert y_train.shape[1] == prediction_horizon

print("Dataset sizes are compatible.")

# ============================================================
# 8. FLATTEN
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

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

# ============================================================
# 9. NORMALIZATION
# ============================================================

X_mean = X_train.mean(
    dim=0,
    keepdim=True
)

X_std = X_train.std(
    dim=0,
    keepdim=True
)

X_std[X_std < 1e-8] = 1.0

X_train_norm = (
    X_train - X_mean
) / X_std

X_test_norm = (
    X_test - X_mean
) / X_std

# ============================================================
# TARGET NORMALIZATION
# ============================================================

y_mean = y_train.mean(
    dim=0,
    keepdim=True
)

y_std = y_train.std(
    dim=0,
    keepdim=True
)

y_std[y_std < 1e-8] = 1.0

y_train_norm = (
    y_train - y_mean
) / y_std

y_test_norm = (
    y_test - y_mean
) / y_std

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
# 10. DATASET
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
# 11. RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        features,
        dropout=0.05
    ):

        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(
                features,
                features
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                features,
                features
            )
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.block(x)

        x = x + residual

        return self.activation(x)


# ============================================================
# 12. MODEL
# ============================================================

class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_size,
        output_size
    ):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(
                input_size,
                256
            ),
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
            nn.Linear(
                256,
                128
            ),
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
# 13. CREATE MODEL
# ============================================================

input_size = X_train_norm.shape[1]
output_size = y_train_norm.shape[1]

model = ResidualMLP(
    input_size,
    output_size
)

print(model)

# ============================================================
# 14. LOSS / OPTIMIZER
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
# 15. TRAIN
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
            loss.item()
            * batch_X.size(0)
        )

    epoch_loss /= len(dataset)

    scheduler.step(epoch_loss)

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
# 16. BEST MODEL
# ============================================================

model.load_state_dict(best_state)

print("Best training loss:", best_loss)

# ============================================================
# 17. TEST
# ============================================================

model.eval()

with torch.no_grad():

    prediction_norm = model(
        X_test_norm.float()
    )

# ============================================================
# 18. DENORMALIZE
# ============================================================

prediction = (
    prediction_norm
    * y_std
    + y_mean
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
# 19. FIRST VALUES
# ============================================================

print("First 10 real values:")
print(real_values[:10])

print("First 10 predicted values:")
print(prediction[:10])

# ============================================================
# 20. OVERALL METRICS
# ============================================================

error = prediction - real_values

mae = torch.abs(error).mean().item()

rmse = torch.sqrt(
    torch.mean(error ** 2)
).item()

print()
print("=================================")
print("OVERALL RESULTS")
print("=================================")

print("MAE:", mae)
print("RMSE:", rmse)

# ============================================================
# 21. HORIZON METRICS
# ============================================================

print()
print("=================================")
print("RESULTS BY HORIZON")
print("=================================")

for h in range(prediction_horizon):

    horizon_error = error[:, h]

    horizon_mae = (
        torch.abs(
            horizon_error
        ).mean().item()
    )

    horizon_rmse = torch.sqrt(
        torch.mean(
            horizon_error ** 2
        )
    ).item()

    print(
        f"Horizon {h + 1}: "
        f"MAE = {horizon_mae:.6f}, "
        f"RMSE = {horizon_rmse:.6f}"
    )

# ============================================================
# 22. BASELINE CHECK
# ============================================================
#
# Important:
# Compare the neural network against a very simple baseline.
#
# Baseline:
# predict every future point as the LAST observed target.
#
# This tells us whether the neural network is actually learning
# the dynamics or simply exploiting the smoothness of the data.
#
# ============================================================

baseline_predictions = []

for traj in range(X_test_raw.shape[0]):

    for t in range(n_samples_per_trajectory):

        last_value = target_test_raw[
            traj,
            t + sequence_length - 1
        ]

        baseline_predictions.append(
            torch.full(
                (prediction_horizon,),
                last_value
            )
        )

baseline_predictions = torch.stack(
    baseline_predictions
).double()

baseline_error = (
    baseline_predictions
    - real_values
)

baseline_mae = torch.abs(
    baseline_error
).mean().item()

baseline_rmse = torch.sqrt(
    torch.mean(
        baseline_error ** 2
    )
).item()

print()
print("=================================")
print("BASELINE RESULTS")
print("=================================")

print(
    "Baseline MAE:",
    baseline_mae
)

print(
    "Baseline RMSE:",
    baseline_rmse
)

# ============================================================
# 23. IMPROVEMENT OVER BASELINE
# ============================================================

mae_improvement = (
    100.0
    * (baseline_mae - mae)
    / baseline_mae
)

rmse_improvement = (
    100.0
    * (baseline_rmse - rmse)
    / baseline_rmse
)

print()
print("=================================")
print("IMPROVEMENT OVER BASELINE")
print("=================================")

print(
    f"MAE improvement: "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)

# ============================================================
# 24. SAVE
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
        "prediction_horizon": prediction_horizon,
        "best_training_loss": best_loss,
        "test_mae": mae,
        "test_rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse
    },
    "model24_best.pt"
)

print()
print("Model saved as model24_best.pt")
print("MODEL 24 COMPLETED.")