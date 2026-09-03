import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

SEED = 42

INPUT_WINDOW = 5
HORIZON = 5

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.001

DATA_PATH = r".\generated_data_ramp_10h\combo_0000.pt"
SAVE_PATH = r".\model26b_best.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

data = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

X = data["X"]
Y = data["Y"]
U = data["U"]

target = Y[:, :, 0].double()

print(dict(data))
print("X:", X.shape)
print("Y:", Y.shape)
print("U:", U.shape)
print("Target shape:", target.shape)


# ============================================================
# TRAJECTORY SPLIT
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

num_val = max(
    1,
    int(num_train_pool * 0.10)
)

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    num_train_pool,
    generator=generator
)

val_indices = indices[:num_val]
train_indices = indices[num_val:]


X_train_raw = X_train_pool[train_indices]
U_train_raw = U_train_pool[train_indices]
Y_train_raw = Y_train_pool[train_indices]

X_val_raw = X_train_pool[val_indices]
U_val_raw = U_train_pool[val_indices]
Y_val_raw = Y_train_pool[val_indices]


print()
print("=================================")
print("TRAJECTORY SPLIT")
print("=================================")

print("Total trajectories:", num_trajectories)
print("Training trajectories:", X_train_raw.shape[0])
print("Validation trajectories:", X_val_raw.shape[0])
print("Testing trajectories:", X_test_raw.shape[0])


# ============================================================
# BUILD DATASET
#
# NEW:
# Add last observed target Y(t-1)
# as ONE additional feature to every timestep.
#
# Original:
# 5 timesteps x 12 features = 60
#
# New:
# 5 timesteps x 13 features = 65
# ============================================================

def build_dataset(X_data, U_data, Y_data):

    features = []
    targets = []

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

            # ------------------------------------------------
            # Last known target Y(t-1)
            # This is known at prediction time.
            # ------------------------------------------------

            last_y = Y_data[
                traj,
                t - 1
            ]

            # Repeat the scalar over the 5 input timesteps.
            last_y_feature = torch.full(
                (INPUT_WINDOW, 1),
                last_y.item(),
                dtype=x_window.dtype
            )

            combined = torch.cat(
                [
                    x_window,
                    u_window,
                    last_y_feature
                ],
                dim=-1
            )

            y_future = Y_data[
                traj,
                t:t + HORIZON
            ]

            features.append(combined)
            targets.append(y_future)

    return (
        torch.stack(features),
        torch.stack(targets)
    )


# ============================================================
# BUILD TRAIN / VAL / TEST WINDOWS
# ============================================================

X_train_seq, y_train = build_dataset(
    X_train_raw,
    U_train_raw,
    Y_train_raw
)

X_val_seq, y_val = build_dataset(
    X_val_raw,
    U_val_raw,
    Y_val_raw
)

X_test_seq, y_test = build_dataset(
    X_test_raw,
    U_test_raw,
    Y_test_raw
)


print()
print("=================================")
print("WINDOW DATASETS")
print("=================================")

print("Training features:", X_train_seq.shape)
print("Training targets:", y_train.shape)

print("Validation features:", X_val_seq.shape)
print("Validation targets:", y_val.shape)

print("Testing features:", X_test_seq.shape)
print("Testing targets:", y_test.shape)


# ============================================================
# FLATTEN
# ============================================================

X_train = X_train_seq.reshape(
    X_train_seq.shape[0],
    -1
)

X_val = X_val_seq.reshape(
    X_val_seq.shape[0],
    -1
)

X_test = X_test_seq.reshape(
    X_test_seq.shape[0],
    -1
)

y_train = y_train.float()
y_val = y_val.float()
y_test = y_test.float()


print()
print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)

print("Final X_val:", X_val.shape)
print("Final y_val:", y_val.shape)

print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)


# ============================================================
# NORMALIZATION
# TRAINING DATA ONLY
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

X_val_norm = (
    X_val - X_mean
) / X_std

X_test_norm = (
    X_test - X_mean
) / X_std


print()
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
# DATALOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_norm.float(),
    y_train
)

val_dataset = TensorDataset(
    X_val_norm.float(),
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


print()
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)


# ============================================================
# MODEL
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

        out = self.block(x)

        out = out + residual

        return self.activation(out)


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
            0.05
        )

        self.residual2 = ResidualBlock(
            256,
            0.05
        )

        self.residual3 = ResidualBlock(
            256,
            0.05
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

        return self.output_layer(x)


# ============================================================
# CREATE MODEL
# ============================================================

INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = HORIZON

model = ResidualMLP(
    INPUT_SIZE,
    OUTPUT_SIZE
)

print()
print(model)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

best_state = None


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        predictions = model(batch_x)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item() * batch_x.size(0)
        )

    train_loss = (
        train_loss_sum /
        len(train_loader.dataset)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            predictions = model(batch_x)

            loss = criterion(
                predictions,
                batch_y
            )

            val_loss_sum += (
                loss.item() * batch_x.size(0)
            )

    val_loss = (
        val_loss_sum /
        len(val_loader.dataset)
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            key: value.cpu().clone()
            for key, value in
            model.state_dict().items()
        }


    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Train Loss: {train_loss:.6f} "
            f"- Val Loss: {val_loss:.6f}"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)

model.eval()


print()
print("Best validation loss:", best_val_loss)


# ============================================================
# TEST
# ============================================================

with torch.no_grad():

    predictions = model(
        X_test_norm.float()
    )

real_values = y_test


print()
print("Test prediction shape:", predictions.shape)
print("Real test values shape:", real_values.shape)


# ============================================================
# METRICS
# ============================================================

absolute_error = (
    predictions - real_values
).abs()

squared_error = (
    predictions - real_values
) ** 2

mae = absolute_error.mean().item()

rmse = torch.sqrt(
    squared_error.mean()
).item()


print()
print("=================================")
print("MODEL 26B TEST RESULTS")
print("=================================")

print(
    f"Overall MAE : {mae:.6f}"
)

print(
    f"Overall RMSE: {rmse:.6f}"
)


# ============================================================
# HORIZON METRICS
# ============================================================

print()
print("Horizon metrics:")

for h in range(HORIZON):

    h_mae = absolute_error[:, h].mean().item()

    h_rmse = torch.sqrt(
        squared_error[:, h].mean()
    ).item()

    print(
        f"H{h+1}: "
        f"MAE = {h_mae:.6f}, "
        f"RMSE = {h_rmse:.6f}"
    )


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

# Correct baseline:
# predict every future horizon with Y(t-1)

last_observed_targets = []

n_test_traj = Y_test_raw.shape[0]
n_test_time = Y_test_raw.shape[1]

for traj in range(n_test_traj):

    for t in range(
        INPUT_WINDOW,
        n_test_time - HORIZON + 1
    ):

        last_observed_targets.append(
            Y_test_raw[
                traj,
                t - 1
            ]
        )

last_observed_targets = torch.stack(
    last_observed_targets
).float()

baseline_predictions = last_observed_targets.unsqueeze(1).repeat(
    1,
    HORIZON
)

baseline_error = (
    baseline_predictions - real_values
).abs()

baseline_squared = (
    baseline_predictions - real_values
) ** 2

baseline_mae = (
    baseline_error.mean().item()
)

baseline_rmse = torch.sqrt(
    baseline_squared.mean()
).item()


print()
print("=================================")
print("PERSISTENCE BASELINE")
print("=================================")

print(
    f"Baseline MAE : {baseline_mae:.6f}"
)

print(
    f"Baseline RMSE: {baseline_rmse:.6f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

mae_improvement = (
    100.0 *
    (baseline_mae - mae)
    / baseline_mae
)

rmse_improvement = (
    100.0 *
    (baseline_rmse - rmse)
    / baseline_rmse
)


print()
print("=================================")
print("MODEL 26B vs BASELINE")
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
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "input_window": INPUT_WINDOW,
        "horizon": HORIZON,
        "best_val_loss": best_val_loss,
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "train_trajectory_indices": train_indices,
        "validation_trajectory_indices": val_indices
    },
    SAVE_PATH
)


print()
print("Saved:", SAVE_PATH)

print()
print("=================================")
print("MODEL 26B COMPLETED")
print("=================================")