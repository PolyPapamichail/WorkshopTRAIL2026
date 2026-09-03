import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

n_trajectories = target.shape[0]
n_train = int(0.80 * n_trajectories)

train_target = target[:n_train]
test_target = target[n_train:]

print("Total trajectories:", n_trajectories)
print("Training trajectories:", train_target.shape)
print("Testing trajectories:", test_target.shape)

inputs = torch.cat([X, U], dim=2)

# Use the previous 5 time steps to predict the next 5 Y values

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

X_train = features_train.reshape(-1, 60)
y_train = target_train.reshape(-1, 5)

X_test = features_test.reshape(-1, 60)
y_test = target_test.reshape(-1, 5)

print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)
print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)

mean = X_train.mean(dim=0)
std = X_train.std(dim=0)

X_train_norm = (
    X_train - mean
) / (std + 1e-8)

X_test_norm = (
    X_test - mean
) / (std + 1e-8)

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

model = nn.Sequential(
    nn.Linear(60, 256),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(256, 128),
    nn.ReLU(),

    nn.Linear(128, 5)
)

print(model)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

for epoch in range(epochs):

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

model.eval()

with torch.no_grad():

    test_prediction_norm = model(
        X_test_norm.float()
    )

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

print("First 10 real values:")
print(y_test[:10])

print("First 10 predicted values:")
print(test_prediction[:10])

error = (
    test_prediction - y_test
)

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

print("MAE:", mae.item())
print("RMSE:", rmse.item())

print()
print("=================================")
print("RESULTS BY HORIZON")
print("=================================")

mae_values = []
rmse_values = []

for h in range(5):

    error_h = (
        test_prediction[:, h]
        - y_test[:, h]
    )

    mae_h = torch.mean(
        torch.abs(error_h)
    )

    rmse_h = torch.sqrt(
        torch.mean(error_h ** 2)
    )

    mae_values.append(
        mae_h.item()
    )

    rmse_values.append(
        rmse_h.item()
    )

    print(
        f"Horizon {h + 1}: "
        f"MAE = {mae_h.item():.6f}, "
        f"RMSE = {rmse_h.item():.6f}"
    )

with open(
    "experiment15_results.txt",
    "w"
) as f:

    f.write(
        "EXPERIMENT 15\n"
    )

    f.write(
        "5 previous time steps -> 5 future Y values\n\n"
    )

    f.write(
        f"Overall MAE: {mae.item():.6f}\n"
    )

    f.write(
        f"Overall RMSE: {rmse.item():.6f}\n\n"
    )

    for h in range(5):

        f.write(
            f"Horizon {h + 1}: "
            f"MAE = {mae_values[h]:.6f}, "
            f"RMSE = {rmse_values[h]:.6f}\n"
        )

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, 6),
    mae_values,
    marker="o"
)

plt.xlabel("Prediction Horizon")
plt.ylabel("MAE")

plt.title(
    "MAE by Prediction Horizon"
)

plt.xticks(
    range(1, 6)
)

plt.grid()
plt.tight_layout()

plt.savefig(
    "experiment15_mae_by_horizon.png",
    dpi=150
)

plt.close()

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, 6),
    rmse_values,
    marker="o"
)

plt.xlabel("Prediction Horizon")
plt.ylabel("RMSE")

plt.title(
    "RMSE by Prediction Horizon"
)

plt.xticks(
    range(1, 6)
)

plt.grid()
plt.tight_layout()

plt.savefig(
    "experiment15_rmse_by_horizon.png",
    dpi=150
)

plt.close()

print()
print("=================================")
print("EXPERIMENT 15 COMPLETED")
print("=================================")

print(
    "Results saved to experiment15_results.txt"
)

print(
    "PLOT SAVED: experiment15_mae_by_horizon.png"
)

print(
    "PLOT SAVED: experiment15_rmse_by_horizon.png"
)