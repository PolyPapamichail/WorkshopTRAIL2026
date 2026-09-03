import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# MODEL 21
# Same MLP as Model 20
# Learning-rate scheduler + 200 epochs
# ============================================================

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

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

target = Y[:, :, 0]

print("Target shape:", target.shape)

# ------------------------------------------------------------
# 2. Train / Test split
# ------------------------------------------------------------

n_trajectories = target.shape[0]

n_train = int(0.80 * n_trajectories)

train_target = target[:n_train]
test_target = target[n_train:]

print("Total trajectories:", n_trajectories)
print("Training trajectories:", train_target.shape)
print("Testing trajectories:", test_target.shape)

# ------------------------------------------------------------
# 3. Inputs = X + U
# ------------------------------------------------------------

inputs = torch.cat([X, U], dim=2)

# 5 previous time steps
# 12 features per time step
# 5 x 12 = 60 input features

features_train = torch.cat([
    inputs[:n_train, 0:-9, :],
    inputs[:n_train, 1:-8, :],
    inputs[:n_train, 2:-7, :],
    inputs[:n_train, 3:-6, :],
    inputs[:n_train, 4:-5, :]
], dim=2)

target_train = torch.stack([
    target[:n_train, 5:-4],
    target[:n_train, 6:-3],
    target[:n_train, 7:-2],
    target[:n_train, 8:-1],
    target[:n_train, 9:]
], dim=2)

features_test = torch.cat([
    inputs[n_train:, 0:-9, :],
    inputs[n_train:, 1:-8, :],
    inputs[n_train:, 2:-7, :],
    inputs[n_train:, 3:-6, :],
    inputs[n_train:, 4:-5, :]
], dim=2)

target_test = torch.stack([
    target[n_train:, 5:-4],
    target[n_train:, 6:-3],
    target[n_train:, 7:-2],
    target[n_train:, 8:-1],
    target[n_train:, 9:]
], dim=2)

print("Training features:", features_train.shape)
print("Training target:", target_train.shape)
print("Testing features:", features_test.shape)
print("Testing target:", target_test.shape)

# ------------------------------------------------------------
# 4. Convert trajectories into individual examples
# ------------------------------------------------------------

X_train = features_train.reshape(-1, 60)
y_train = target_train.reshape(-1, 5)

X_test = features_test.reshape(-1, 60)
y_test = target_test.reshape(-1, 5)

print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)
print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)

# ------------------------------------------------------------
# 5. Safety checks
# ------------------------------------------------------------

assert X_train.shape[0] == y_train.shape[0], \
    "Training X and y have different number of samples!"

assert X_test.shape[0] == y_test.shape[0], \
    "Testing X and y have different number of samples!"

print("Dataset sizes are compatible.")

# ------------------------------------------------------------
# 6. Normalize inputs using training data only
# ------------------------------------------------------------

mean = X_train.mean(dim=0)
std = X_train.std(dim=0)

X_train_norm = (
    X_train - mean
) / (std + 1e-8)

X_test_norm = (
    X_test - mean
) / (std + 1e-8)

# ------------------------------------------------------------
# 7. Normalize target
# ------------------------------------------------------------

target_mean = y_train.mean()
target_std = y_train.std()

y_train_norm = (
    y_train - target_mean
) / (target_std + 1e-8)

y_test_norm = (
    y_test - target_mean
) / (target_std + 1e-8)

print("Normalization completed.")

print(
    "Normalized X_train mean:",
    X_train_norm.mean().item()
)

print(
    "Normalized X_train std:",
    X_train_norm.std().item()
)

# ------------------------------------------------------------
# 8. Model
# ------------------------------------------------------------

model = nn.Sequential(
    nn.Linear(60, 256),
    nn.ReLU(),

    nn.Linear(256, 256),
    nn.ReLU(),

    nn.Linear(256, 128),
    nn.ReLU(),

    nn.Linear(128, 5)
)

print(model)

# ------------------------------------------------------------
# 9. Loss and optimizer
# ------------------------------------------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ------------------------------------------------------------
# 10. Learning-rate scheduler
# ------------------------------------------------------------

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10
)

# ------------------------------------------------------------
# 11. Mini-batch DataLoader
# ------------------------------------------------------------

dataset = TensorDataset(
    X_train_norm.float(),
    y_train_norm.float()
)

batch_size = 256

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

print("Training samples:", len(dataset))
print("Batch size:", batch_size)

# ------------------------------------------------------------
# 12. Training
# ------------------------------------------------------------

epochs = 200

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for batch_X, batch_y in train_loader:

        prediction = model(batch_X)

        loss = loss_function(
            prediction,
            batch_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * batch_X.size(0)
        )

    epoch_loss = (
        running_loss
        / len(dataset)
    )

    # Update learning rate
    scheduler.step(epoch_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {epoch_loss:.6f} "
            f"- LR: {current_lr:.6f}"
        )

# ------------------------------------------------------------
# 13. Evaluation
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    test_prediction_norm = model(
        X_test_norm.float()
    )

# ------------------------------------------------------------
# 14. Convert predictions back to original units
# ------------------------------------------------------------

test_prediction = (
    test_prediction_norm
    * (target_std + 1e-8)
    + target_mean
)

print(
    "Test prediction shape:",
    test_prediction.shape
)

print(
    "Real test values shape:",
    y_test.shape
)

# ------------------------------------------------------------
# 15. First 10 predictions
# ------------------------------------------------------------

print("First 10 real values:")

print(
    y_test[:10]
)

print("First 10 predicted values:")

print(
    test_prediction[:10]
)

# ------------------------------------------------------------
# 16. Overall evaluation
# ------------------------------------------------------------

error = test_prediction - y_test

mae = torch.mean(
    torch.abs(error)
)

rmse = torch.sqrt(
    torch.mean(error ** 2)
)

print()
print("=================================")
print("OVERALL RESULTS")
print("=================================")

print(
    "MAE:",
    mae.item()
)

print(
    "RMSE:",
    rmse.item()
)

# ------------------------------------------------------------
# 17. Results by horizon
# ------------------------------------------------------------

print()
print("=================================")
print("RESULTS BY HORIZON")
print("=================================")

for h in range(5):

    horizon_error = (
        test_prediction[:, h]
        - y_test[:, h]
    )

    horizon_mae = torch.mean(
        torch.abs(horizon_error)
    )

    horizon_rmse = torch.sqrt(
        torch.mean(
            horizon_error ** 2
        )
    )

    print(
        f"Horizon {h + 1}: "
        f"MAE = {horizon_mae.item():.6f}, "
        f"RMSE = {horizon_rmse.item():.6f}"
    )

# ------------------------------------------------------------
# 18. Plot first test trajectory
# ------------------------------------------------------------

trajectory = 0

n_points_per_trajectory = 92

start = (
    trajectory
    * n_points_per_trajectory
)

end = (
    (trajectory + 1)
    * n_points_per_trajectory
)

real = (
    y_test[start:end]
    .flatten()
    .detach()
    .numpy()
)

predicted = (
    test_prediction[start:end]
    .flatten()
    .detach()
    .numpy()
)

plt.figure(figsize=(10, 5))

plt.plot(
    real,
    label="Real"
)

plt.plot(
    predicted,
    label="Predicted"
)

plt.xlabel("Time step")

plt.ylabel("Target")

plt.title(
    "Real vs Predicted - "
    "Model 21 - Test Trajectory 0"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "prediction_plot_model21.png",
    dpi=150
)

plt.close()

print()
print("=================================")
print("PLOT SAVED:")
print("prediction_plot_model21.png")
print("=================================")