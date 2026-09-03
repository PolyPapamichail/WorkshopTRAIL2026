import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 11
# 3 previous time steps -> 5 future Y values
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


# ------------------------------------------------------------
# 2. Select target variable
# ------------------------------------------------------------

target = Y[:, :, 0]

print("Target shape:", target.shape)

n_trajectories = target.shape[0]

# 80% training / 20% testing
n_train = int(0.80 * n_trajectories)

train_target = target[:n_train]
test_target = target[n_train:]

print("Total trajectories:", n_trajectories)
print("Training trajectories:", train_target.shape)
print("Testing trajectories:", test_target.shape)


# ------------------------------------------------------------
# 3. Create input features
# ------------------------------------------------------------

# Inputs = X + U
inputs = torch.cat([X, U], dim=2)

# We use:
#
# t-2, t-1, t
#
# to predict:
#
# t+1, t+2, t+3, t+4, t+5

features_train = torch.cat([
    inputs[:n_train, 0:-7, :],
    inputs[:n_train, 1:-6, :],
    inputs[:n_train, 2:-5, :]
], dim=2)

target_train = torch.stack([
    target[:n_train, 3:-4],
    target[:n_train, 4:-3],
    target[:n_train, 5:-2],
    target[:n_train, 6:-1],
    target[:n_train, 7:]
], dim=2)


features_test = torch.cat([
    inputs[n_train:, 0:-7, :],
    inputs[n_train:, 1:-6, :],
    inputs[n_train:, 2:-5, :]
], dim=2)

target_test = torch.stack([
    target[n_train:, 3:-4],
    target[n_train:, 4:-3],
    target[n_train:, 5:-2],
    target[n_train:, 6:-1],
    target[n_train:, 7:]
], dim=2)


print("Training features:", features_train.shape)
print("Training target:", target_train.shape)
print("Testing features:", features_test.shape)
print("Testing target:", target_test.shape)


# ------------------------------------------------------------
# 4. Convert trajectories into individual examples
# ------------------------------------------------------------

X_train = features_train.reshape(-1, 36)
y_train = target_train.reshape(-1, 5)

X_test = features_test.reshape(-1, 36)
y_test = target_test.reshape(-1, 5)

print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)
print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)


# ------------------------------------------------------------
# 5. Normalize inputs
# ------------------------------------------------------------

mean = X_train.mean(dim=0)
std = X_train.std(dim=0)

X_train_norm = (X_train - mean) / (std + 1e-8)
X_test_norm = (X_test - mean) / (std + 1e-8)


# ------------------------------------------------------------
# 6. Normalize target
# ------------------------------------------------------------

target_mean = y_train.mean()
target_std = y_train.std()

y_train_norm = (
    (y_train - target_mean)
    / (target_std + 1e-8)
)

y_test_norm = (
    (y_test - target_mean)
    / (target_std + 1e-8)
)

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
# 7. Neural network
# ------------------------------------------------------------

model = nn.Sequential(
    nn.Linear(36, 128),
    nn.ReLU(),

    nn.Linear(128, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 5)
)

print(model)


# ------------------------------------------------------------
# 8. Training setup
# ------------------------------------------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100


# ------------------------------------------------------------
# 9. Training
# ------------------------------------------------------------

for epoch in range(epochs):

    model.train()

    prediction = model(
        X_train_norm.float()
    )

    loss = loss_function(
        prediction,
        y_train_norm.float()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {loss.item():.6f}"
        )


# ------------------------------------------------------------
# 10. Test prediction
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    test_prediction_norm = model(
        X_test_norm.float()
    )


# ------------------------------------------------------------
# 11. Convert predictions back to original units
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
# 12. Show first predictions
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
# 13. Evaluation metrics
# ------------------------------------------------------------

error = test_prediction - y_test

mae = torch.mean(
    torch.abs(error)
)

rmse = torch.sqrt(
    torch.mean(error ** 2)
)

print("MAE:", mae.item())
print("RMSE:", rmse.item())


# ------------------------------------------------------------
# 14. Plot first test trajectory
# ------------------------------------------------------------

trajectory = 0

# Each trajectory contains 94 valid prediction points
prediction_steps_per_trajectory = target_test.shape[1]

start = trajectory * prediction_steps_per_trajectory
end = (trajectory + 1) * prediction_steps_per_trajectory

real = (
    y_test[start:end]
    .detach()
    .numpy()
)

predicted = (
    test_prediction[start:end]
    .detach()
    .numpy()
)


# ------------------------------------------------------------
# 15. Plot each of the 5 prediction horizons
# ------------------------------------------------------------

for horizon in range(5):

    plt.figure(figsize=(10, 5))

    plt.plot(
        real[:, horizon],
        label="Real"
    )

    plt.plot(
        predicted[:, horizon],
        label="Predicted"
    )

    plt.xlabel("Prediction step")
    plt.ylabel("Target")

    plt.title(
        f"Experiment 11 - Horizon {horizon + 1}"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    filename = (
        f"experiment11_horizon_{horizon + 1}.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    print(
        "PLOT SAVED:",
        filename
    )


# ------------------------------------------------------------
# 16. Save combined plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    real[:, 0],
    label="Real horizon 1"
)

plt.plot(
    predicted[:, 0],
    label="Predicted horizon 1"
)

plt.xlabel("Prediction step")
plt.ylabel("Target")

plt.title(
    "Experiment 11 - Real vs Predicted"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    "experiment11_prediction_plot.png",
    dpi=150
)

plt.close()

print("=================================")
print("EXPERIMENT 11 COMPLETED")
print("MAE:", mae.item())
print("RMSE:", rmse.item())
print("=================================")