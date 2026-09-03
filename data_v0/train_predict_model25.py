import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. LOAD DATA
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
# 3. TARGET
# ============================================================

target = Y[:, :, 0]

print("Target shape:", target.shape)


# ============================================================
# 4. TRAIN / TEST TRAJECTORIES
# ============================================================

num_trajectories = X.shape[0]

train_trajectories = int(0.8 * num_trajectories)

X_train_raw = X[:train_trajectories]
X_test_raw = X[train_trajectories:]

U_train_raw = U[:train_trajectories]
U_test_raw = U[train_trajectories:]

Y_train_raw = target[:train_trajectories]
Y_test_raw = target[train_trajectories:]

print("Total trajectories:", num_trajectories)
print("Training trajectories:", Y_train_raw.shape)
print("Testing trajectories:", Y_test_raw.shape)


# ============================================================
# 5. FEATURES
# ============================================================

num_x_features = X.shape[-1]
num_u_features = U.shape[-1]

print("X features:", num_x_features)
print("U features:", num_u_features)
print("Total features per timestep:", num_x_features + num_u_features)


# ============================================================
# 6. BUILD MULTI-HORIZON DATASET
# ============================================================

INPUT_WINDOW = 5
HORIZON = 5

def build_dataset(X_data, U_data, Y_data):

    features = []
    targets = []

    n_traj = X_data.shape[0]
    n_time = X_data.shape[1]

    for traj in range(n_traj):

        for t in range(INPUT_WINDOW, n_time - HORIZON + 1):

            x_window = X_data[traj, t-INPUT_WINDOW:t, :]
            u_window = U_data[traj, t-INPUT_WINDOW:t, :]

            combined = torch.cat(
                [x_window, u_window],
                dim=-1
            )

            y_future = Y_data[
                traj,
                t:t+HORIZON
            ]

            features.append(combined)
            targets.append(y_future)

    return (
        torch.stack(features),
        torch.stack(targets)
    )


X_train_seq, y_train = build_dataset(
    X_train_raw,
    U_train_raw,
    Y_train_raw
)

X_test_seq, y_test = build_dataset(
    X_test_raw,
    U_test_raw,
    Y_test_raw
)

print("Training features:", X_train_seq.shape)
print("Training target:", y_train.shape)

print("Testing features:", X_test_seq.shape)
print("Testing target:", y_test.shape)


# ============================================================
# 7. FLATTEN INPUT
# ============================================================

X_train = X_train_seq.reshape(
    X_train_seq.shape[0],
    -1
)

X_test = X_test_seq.reshape(
    X_test_seq.shape[0],
    -1
)

print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)

print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)


# ============================================================
# 8. CHECK DATASET SIZES
# ============================================================

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

print("Dataset sizes are compatible.")


# ============================================================
# 9. NORMALIZATION
# ============================================================

X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True)

X_std[X_std < 1e-8] = 1.0

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

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
# 10. TRAIN / VALIDATION SPLIT
# ============================================================

dataset = TensorDataset(
    X_train_norm.float(),
    y_train.float()
)

validation_fraction = 0.15

n_total = len(dataset)
n_val = int(n_total * validation_fraction)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# 11. DATALOADERS
# ============================================================

BATCH_SIZE = 256

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

print("Batch size:", BATCH_SIZE)


# ============================================================
# 12. RESIDUAL BLOCK
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


# ============================================================
# 13. RESIDUAL MLP
# ============================================================

class ResidualMLP(nn.Module):

    def __init__(self, input_size, output_size):

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

        return self.output_layer(x)


# ============================================================
# 14. MODEL
# ============================================================

INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = HORIZON

model = ResidualMLP(
    INPUT_SIZE,
    OUTPUT_SIZE
)

print(model)


# ============================================================
# 15. LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-6
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=12
)


# ============================================================
# 16. TRAINING WITH VALIDATION + EARLY STOPPING
# ============================================================

MAX_EPOCHS = 300
PATIENCE = 35

best_val_loss = float("inf")
best_state = None

epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        prediction = model(xb)

        loss = criterion(
            prediction,
            yb
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)
        train_count += xb.size(0)

    train_loss = train_loss_sum / train_count


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            prediction = model(xb)

            loss = criterion(
                prediction,
                yb
            )

            val_loss_sum += loss.item() * xb.size(0)
            val_count += xb.size(0)

    val_loss = val_loss_sum / val_count

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if epoch % 10 == 0 or epoch == 1:

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} "
            f"- Train Loss: {train_loss:.6f} "
            f"- Val Loss: {val_loss:.6f} "
            f"- LR: {current_lr:.6f}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"Early stopping at epoch {epoch}"
        )

        break


# ============================================================
# 17. RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(best_state)

print(
    "Best validation loss:",
    best_val_loss
)


# ============================================================
# 18. TEST PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    predictions_norm = model(
        X_test_norm.float()
    )


# ============================================================
# 19. PREDICTIONS ARE TARGET VALUES
# ============================================================

predictions = predictions_norm.double()

real_values = y_test.double()

print(
    "Test prediction shape:",
    predictions.shape
)

print(
    "Real test values shape:",
    real_values.shape
)


# ============================================================
# 20. FIRST 10 PREDICTIONS
# ============================================================

print("First 10 real values:")
print(real_values[:10])

print("First 10 predicted values:")
print(predictions[:10])


# ============================================================
# 21. OVERALL METRICS
# ============================================================

absolute_error = torch.abs(
    predictions - real_values
)

squared_error = (
    predictions - real_values
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
# 22. RESULTS BY HORIZON
# ============================================================

print()
print("=================================")
print("RESULTS BY HORIZON")
print("=================================")

for h in range(HORIZON):

    horizon_mae = torch.mean(
        torch.abs(
            predictions[:, h] -
            real_values[:, h]
        )
    ).item()

    horizon_rmse = torch.sqrt(
        torch.mean(
            (
                predictions[:, h] -
                real_values[:, h]
            ) ** 2
        )
    ).item()

    print(
        f"Horizon {h+1}: "
        f"MAE = {horizon_mae:.6f}, "
        f"RMSE = {horizon_rmse:.6f}"
    )


# ============================================================
# 23. BASELINE
# ============================================================

baseline_predictions = X_test_seq[:, -1, 0].unsqueeze(1).repeat(
    1,
    HORIZON
).double()

baseline_mae = torch.mean(
    torch.abs(
        baseline_predictions -
        real_values
    )
).item()

baseline_rmse = torch.sqrt(
    torch.mean(
        (
            baseline_predictions -
            real_values
        ) ** 2
    )
).item()


print()
print("=================================")
print("BASELINE RESULTS")
print("=================================")

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)


# ============================================================
# 24. IMPROVEMENT
# ============================================================

mae_improvement = (
    (baseline_mae - mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
) * 100


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
# 25. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "input_window": INPUT_WINDOW,
        "horizon": HORIZON,
        "best_val_loss": best_val_loss,
        "mae": mae,
        "rmse": rmse
    },
    "model25_best.pt"
)

print()
print("Model saved as model25_best.pt")
print("MODEL 25 COMPLETED.")