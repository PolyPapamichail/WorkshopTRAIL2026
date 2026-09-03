import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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
# 3. TRAIN / TEST SPLIT
# ============================================================

n_total = target.shape[0]
n_train = int(0.8 * n_total)

X_train_raw = X[:n_train]
X_test_raw = X[n_train:]

U_train_raw = U[:n_train]
U_test_raw = U[n_train:]

target_train = target[:n_train]
target_test = target[n_train:]

print("Total trajectories:", n_total)
print("Training trajectories:", target_train.shape)
print("Testing trajectories:", target_test.shape)


# ============================================================
# 4. CREATE SEQUENCES
# ============================================================

sequence_length = 5
forecast_horizon = 5

X_train = []
y_train = []

X_test = []
y_test = []


# ============================================================
# TRAINING SEQUENCES
# ============================================================

for i in range(n_train):

    for t in range(
        sequence_length,
        target_train.shape[1] - forecast_horizon + 1
    ):

        x_seq = X_train_raw[
            i,
            t - sequence_length:t,
            :
        ]

        u_seq = U_train_raw[
            i,
            t - sequence_length:t,
            :
        ]

        features = torch.cat(
            [
                x_seq,
                u_seq
            ],
            dim=1
        )

        future_target = target_train[
            i,
            t:t + forecast_horizon
        ]

        X_train.append(features)
        y_train.append(future_target)


# ============================================================
# TESTING SEQUENCES
# ============================================================

for i in range(target_test.shape[0]):

    for t in range(
        sequence_length,
        target_test.shape[1] - forecast_horizon + 1
    ):

        x_seq = X_test_raw[
            i,
            t - sequence_length:t,
            :
        ]

        u_seq = U_test_raw[
            i,
            t - sequence_length:t,
            :
        ]

        features = torch.cat(
            [
                x_seq,
                u_seq
            ],
            dim=1
        )

        future_target = target_test[
            i,
            t:t + forecast_horizon
        ]

        X_test.append(features)
        y_test.append(future_target)


X_train = torch.stack(X_train)
y_train = torch.stack(y_train)

X_test = torch.stack(X_test)
y_test = torch.stack(y_test)


print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

print("Testing features:", X_test.shape)
print("Testing target:", y_test.shape)


# ============================================================
# 5. CHECK DATASET
# ============================================================

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

print("Dataset sizes are compatible.")


# ============================================================
# 6. NORMALIZATION
# ============================================================

X_mean = X_train.mean(
    dim=(0, 1),
    keepdim=True
)

X_std = X_train.std(
    dim=(0, 1),
    keepdim=True
)

X_std[X_std == 0] = 1.0


X_train_norm = (
    X_train - X_mean
) / X_std


X_test_norm = (
    X_test - X_mean
) / X_std


y_mean = y_train.mean(
    dim=0,
    keepdim=True
)

y_std = y_train.std(
    dim=0,
    keepdim=True
)

y_std[y_std == 0] = 1.0


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
# 7. MODEL
# ============================================================

class SequenceMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(5 * 12, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 5)
        )

    def forward(self, x):

        x = x.reshape(
            x.shape[0],
            -1
        )

        return self.network(x)


model = SequenceMLP()

print(model)


# ============================================================
# 8. DATASET
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
# 9. LOSS / OPTIMIZER
# ============================================================

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10
)


# ============================================================
# 10. TRAINING
# ============================================================

epochs = 200

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = loss_function(
            prediction,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch_X.size(0)
        )

    epoch_loss = (
        total_loss
        / len(dataset)
    )

    scheduler.step(epoch_loss)

    if (epoch + 1) % 10 == 0:

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {epoch_loss:.6f} "
            f"- LR: {current_lr:.6f}"
        )


# ============================================================
# 11. TEST
# ============================================================

model.eval()

with torch.no_grad():

    prediction_norm = model(
        X_test_norm.float()
    )


# ============================================================
# 12. DENORMALIZATION
# ============================================================

prediction = (
    prediction_norm
    * y_std
    + y_mean
)

prediction = prediction.double()

y_test_real = y_test.double()


print(
    "Test prediction shape:",
    prediction.shape
)

print(
    "Real test values shape:",
    y_test_real.shape
)


# ============================================================
# 13. FIRST 10 VALUES
# ============================================================

print("First 10 real values:")

print(
    y_test_real[:10]
)

print("First 10 predicted values:")

print(
    prediction[:10]
)


# ============================================================
# 14. OVERALL METRICS
# ============================================================

mae = torch.mean(
    torch.abs(
        prediction
        - y_test_real
    )
)

rmse = torch.sqrt(
    torch.mean(
        (
            prediction
            - y_test_real
        ) ** 2
    )
)


print()
print("=================================")
print("OVERALL RESULTS")
print("=================================")

print(
    f"MAE: {mae.item()}"
)

print(
    f"RMSE: {rmse.item()}"
)


# ============================================================
# 15. RESULTS BY HORIZON
# ============================================================

print()
print("=================================")
print("RESULTS BY HORIZON")
print("=================================")


for h in range(forecast_horizon):

    mae_h = torch.mean(
        torch.abs(
            prediction[:, h]
            - y_test_real[:, h]
        )
    )

    rmse_h = torch.sqrt(
        torch.mean(
            (
                prediction[:, h]
                - y_test_real[:, h]
            ) ** 2
        )
    )

    print(
        f"Horizon {h + 1}: "
        f"MAE = {mae_h.item():.6f}, "
        f"RMSE = {rmse_h.item():.6f}"
    )