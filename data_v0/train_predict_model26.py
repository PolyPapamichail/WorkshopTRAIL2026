import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


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
# 4. TRAIN / VALIDATION / TEST TRAJECTORIES
#
# First:
#   80% of trajectories -> training pool
#   20% of trajectories -> completely independent test
#
# Then:
#   90% of training pool -> actual training trajectories
#   10% of training pool -> validation trajectories
#
# IMPORTANT:
# Validation is split BEFORE window creation.
# Therefore, no overlapping windows from the same trajectory
# can appear in both training and validation.
# ============================================================

num_trajectories = X.shape[0]

train_pool_size = int(
    0.8 * num_trajectories
)

# ------------------------------------------------------------
# Independent test trajectories
# ------------------------------------------------------------

X_train_pool = X[:train_pool_size]
U_train_pool = U[:train_pool_size]
Y_train_pool = target[:train_pool_size]

X_test_raw = X[train_pool_size:]
U_test_raw = U[train_pool_size:]
Y_test_raw = target[train_pool_size:]


# ------------------------------------------------------------
# Trajectory-level TRAIN / VALIDATION split
# ------------------------------------------------------------

validation_fraction = 0.10

num_train_pool_trajectories = (
    X_train_pool.shape[0]
)

num_val_trajectories = max(
    1,
    int(
        num_train_pool_trajectories
        * validation_fraction
    )
)

generator = torch.Generator().manual_seed(
    SEED
)

trajectory_indices = torch.randperm(
    num_train_pool_trajectories,
    generator=generator
)

val_indices = trajectory_indices[
    :num_val_trajectories
]

train_indices = trajectory_indices[
    num_val_trajectories:
]


X_train_raw = X_train_pool[
    train_indices
]

U_train_raw = U_train_pool[
    train_indices
]

Y_train_raw = Y_train_pool[
    train_indices
]


X_val_raw = X_train_pool[
    val_indices
]

U_val_raw = U_train_pool[
    val_indices
]

Y_val_raw = Y_train_pool[
    val_indices
]


print()
print("=================================")
print("TRAJECTORY SPLIT")
print("=================================")

print(
    "Total trajectories:",
    num_trajectories
)

print(
    "Training trajectories:",
    X_train_raw.shape[0]
)

print(
    "Validation trajectories:",
    X_val_raw.shape[0]
)

print(
    "Testing trajectories:",
    X_test_raw.shape[0]
)


# ============================================================
# 5. CHECK X / U FEATURES
# ============================================================

num_x_features = X.shape[-1]
num_u_features = U.shape[-1]

print()
print("X features:", num_x_features)
print("U features:", num_u_features)

print(
    "Total features per timestep:",
    num_x_features + num_u_features
)

assert num_x_features == 8
assert num_u_features == 4


# ============================================================
# 6. DATASET PARAMETERS
# ============================================================

INPUT_WINDOW = 5
HORIZON = 5


# ============================================================
# 7. BUILD DATASET
#
# Input:
#   t-5, t-4, t-3, t-2, t-1
#
# Target:
#   t, t+1, t+2, t+3, t+4
#
# We also return Y(t-1), which is needed for the
# persistence baseline.
# ============================================================

def build_dataset(
    X_data,
    U_data,
    Y_data
):

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

            # ------------------------------------------------
            # Past state window
            # ------------------------------------------------

            x_window = X_data[
                traj,
                t - INPUT_WINDOW:t,
                :
            ]

            # ------------------------------------------------
            # Past control window
            # ------------------------------------------------

            u_window = U_data[
                traj,
                t - INPUT_WINDOW:t,
                :
            ]

            # ------------------------------------------------
            # Combine X and U
            # Shape: [INPUT_WINDOW, 12]
            # ------------------------------------------------

            combined = torch.cat(
                [
                    x_window,
                    u_window
                ],
                dim=-1
            )

            # ------------------------------------------------
            # Future target
            # ------------------------------------------------

            y_future = Y_data[
                traj,
                t:t + HORIZON
            ]

            # ------------------------------------------------
            # Last target actually observed before prediction
            #
            # This is Y(t-1).
            # ------------------------------------------------

            last_y = Y_data[
                traj,
                t - 1
            ]

            features.append(
                combined
            )

            targets.append(
                y_future
            )

            last_observed_targets.append(
                last_y
            )

    return (
        torch.stack(features),
        torch.stack(targets),
        torch.stack(last_observed_targets)
    )


# ============================================================
# 8. CREATE TRAIN / VALIDATION / TEST WINDOWS
# ============================================================

(
    X_train_seq,
    y_train,
    train_last_y
) = build_dataset(
    X_train_raw,
    U_train_raw,
    Y_train_raw
)


(
    X_val_seq,
    y_val,
    val_last_y
) = build_dataset(
    X_val_raw,
    U_val_raw,
    Y_val_raw
)


(
    X_test_seq,
    y_test,
    test_last_y
) = build_dataset(
    X_test_raw,
    U_test_raw,
    Y_test_raw
)


print()
print("=================================")
print("WINDOW DATASETS")
print("=================================")

print(
    "Training features:",
    X_train_seq.shape
)

print(
    "Training targets:",
    y_train.shape
)

print(
    "Validation features:",
    X_val_seq.shape
)

print(
    "Validation targets:",
    y_val.shape
)

print(
    "Testing features:",
    X_test_seq.shape
)

print(
    "Testing targets:",
    y_test.shape
)


# ============================================================
# 9. FLATTEN INPUT WINDOWS
#
# [samples, 5, 12]
# ->
# [samples, 60]
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


print()
print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)

print("Final X_val:", X_val.shape)
print("Final y_val:", y_val.shape)

print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)


# ============================================================
# 10. SHAPE CHECKS
# ============================================================

assert (
    X_train.shape[0]
    == y_train.shape[0]
)

assert (
    X_val.shape[0]
    == y_val.shape[0]
)

assert (
    X_test.shape[0]
    == y_test.shape[0]
)

assert (
    X_train.shape[1]
    == INPUT_WINDOW * 12
)

assert (
    X_val.shape[1]
    == INPUT_WINDOW * 12
)

assert (
    X_test.shape[1]
    == INPUT_WINDOW * 12
)

assert (
    y_train.shape[1]
    == HORIZON
)

assert (
    y_val.shape[1]
    == HORIZON
)

assert (
    y_test.shape[1]
    == HORIZON
)

print(
    "Dataset sizes are compatible."
)


# ============================================================
# 11. NORMALIZATION
#
# CRITICAL:
# Mean and std are computed ONLY from the actual training
# trajectories/windows.
#
# Validation and test use those training statistics.
# ============================================================

X_mean = X_train.mean(
    dim=0,
    keepdim=True
)

X_std = X_train.std(
    dim=0,
    keepdim=True
)

X_std[
    X_std < 1e-8
] = 1.0


X_train_norm = (
    X_train
    - X_mean
) / X_std


X_val_norm = (
    X_val
    - X_mean
) / X_std


X_test_norm = (
    X_test
    - X_mean
) / X_std


print()
print(
    "Normalization completed."
)

print(
    "Normalized X_train mean:",
    X_train_norm.mean().item()
)

print(
    "Normalized X_train std:",
    X_train_norm.std().item()
)


# ============================================================
# 12. DATASETS
# ============================================================

train_dataset = TensorDataset(
    X_train_norm.float(),
    y_train.float()
)

val_dataset = TensorDataset(
    X_val_norm.float(),
    y_val.float()
)


print()
print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# 13. DATALOADERS
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

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# 14. RESIDUAL BLOCK
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

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                features,
                features
            )
        )

        self.activation = nn.ReLU()


    def forward(
        self,
        x
    ):

        residual = x

        out = self.block(
            x
        )

        out = (
            out
            + residual
        )

        return self.activation(
            out
        )


# ============================================================
# 15. RESIDUAL MLP
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


    def forward(
        self,
        x
    ):

        x = self.input_layer(
            x
        )

        x = self.residual1(
            x
        )

        x = self.residual2(
            x
        )

        x = self.residual3(
            x
        )

        x = self.hidden(
            x
        )

        return self.output_layer(
            x
        )


# ============================================================
# 16. MODEL
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
# 17. LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0
)


# ============================================================
# 18. TRAINING
# ============================================================

EPOCHS = 200

best_val_loss = float(
    "inf"
)

best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        prediction = model(
            xb
        )

        loss = criterion(
            prediction,
            yb
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item()
            * xb.size(0)
        )

        train_count += xb.size(0)


    train_loss = (
        train_loss_sum
        / train_count
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            prediction = model(
                xb
            )

            loss = criterion(
                prediction,
                yb
            )

            val_loss_sum += (
                loss.item()
                * xb.size(0)
            )

            val_count += xb.size(0)


    val_loss = (
        val_loss_sum
        / val_count
    )


    # --------------------------------------------------------
    # SAVE BEST VALIDATION MODEL
    # --------------------------------------------------------

    if (
        val_loss
        < best_val_loss
    ):

        best_val_loss = (
            val_loss
        )

        best_state = copy.deepcopy(
            model.state_dict()
        )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if (
        epoch % 10 == 0
        or epoch == 1
    ):

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Train Loss: {train_loss:.6f} "
            f"- Val Loss: {val_loss:.6f}"
        )


# ============================================================
# 19. RESTORE BEST VALIDATION MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


print()
print(
    "Best validation loss:",
    best_val_loss
)


# ============================================================
# 20. TEST
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_test_norm.float()
    )


predictions = predictions.double()

real_values = y_test.double()


print()
print(
    "Test prediction shape:",
    predictions.shape
)

print(
    "Real test values shape:",
    real_values.shape
)


# ============================================================
# 21. FIRST 10 VALUES
# ============================================================

print()
print(
    "First 10 real values:"
)

print(
    real_values[:10]
)

print()
print(
    "First 10 predicted values:"
)

print(
    predictions[:10]
)


# ============================================================
# 22. OVERALL RESULTS
# ============================================================

errors = (
    predictions
    - real_values
)

mae = torch.mean(
    torch.abs(
        errors
    )
).item()

rmse = torch.sqrt(
    torch.mean(
        errors ** 2
    )
).item()


print()
print(
    "================================="
)

print(
    "OVERALL RESULTS"
)

print(
    "================================="
)

print(
    "MAE:",
    mae
)

print(
    "RMSE:",
    rmse
)


# ============================================================
# 23. RESULTS BY HORIZON
# ============================================================

print()
print(
    "================================="
)

print(
    "RESULTS BY HORIZON"
)

print(
    "================================="
)


for h in range(
    HORIZON
):

    horizon_errors = (
        predictions[:, h]
        - real_values[:, h]
    )

    horizon_mae = torch.mean(
        torch.abs(
            horizon_errors
        )
    ).item()

    horizon_rmse = torch.sqrt(
        torch.mean(
            horizon_errors ** 2
        )
    ).item()

    print(
        f"Horizon {h + 1}: "
        f"MAE = {horizon_mae:.6f}, "
        f"RMSE = {horizon_rmse:.6f}"
    )


# ============================================================
# 24. CORRECT PERSISTENCE BASELINE
#
# Prediction:
#
# Y(t), Y(t+1), ..., Y(t+4)
#
# are all predicted as:
#
# Y(t-1)
#
# i.e. the last target value actually observed before
# the forecast begins.
# ============================================================

baseline_value = (
    test_last_y
    .double()
    .unsqueeze(1)
)

baseline_predictions = (
    baseline_value.repeat(
        1,
        HORIZON
    )
)


baseline_errors = (
    baseline_predictions
    - real_values
)


baseline_mae = torch.mean(
    torch.abs(
        baseline_errors
    )
).item()


baseline_rmse = torch.sqrt(
    torch.mean(
        baseline_errors ** 2
    )
).item()


print()
print(
    "================================="
)

print(
    "BASELINE RESULTS"
)

print(
    "================================="
)

print(
    "Baseline MAE:",
    baseline_mae
)

print(
    "Baseline RMSE:",
    baseline_rmse
)


# ============================================================
# 25. IMPROVEMENT OVER BASELINE
# ============================================================

mae_improvement = (
    (
        baseline_mae
        - mae
    )
    / baseline_mae
) * 100


rmse_improvement = (
    (
        baseline_rmse
        - rmse
    )
    / baseline_rmse
) * 100


print()
print(
    "================================="
)

print(
    "IMPROVEMENT OVER BASELINE"
)

print(
    "================================="
)

print(
    f"MAE improvement: "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)


# ============================================================
# 26. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "X_mean":
            X_mean,

        "X_std":
            X_std,

        "input_size":
            INPUT_SIZE,

        "output_size":
            OUTPUT_SIZE,

        "input_window":
            INPUT_WINDOW,

        "horizon":
            HORIZON,

        "best_val_loss":
            best_val_loss,

        "mae":
            mae,

        "rmse":
            rmse,

        "baseline_mae":
            baseline_mae,

        "baseline_rmse":
            baseline_rmse,

        "train_trajectory_indices":
            train_indices,

        "validation_trajectory_indices":
            val_indices
    },
    "model26_best.pt"
)


print()
print(
    "Model saved as model26_best.pt"
)

print(
    "MODEL 26 COMPLETED."
)