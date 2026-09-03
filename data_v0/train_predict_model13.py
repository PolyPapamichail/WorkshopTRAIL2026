import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 13
# LSTM
# 5 previous time steps -> 5 future Y values
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
# 3. Prepare sequential input
# ------------------------------------------------------------

# X = 8 variables
# U = 4 variables
# Total input variables = 12
#
# We use 5 consecutive time steps
# to predict the next 5 Y values.


inputs = torch.cat([X, U], dim=2)

window = 5
horizon = 5

X_train_sequences = []
y_train_sequences = []

X_test_sequences = []
y_test_sequences = []


# ------------------------------------------------------------
# 4. Create training sequences
# ------------------------------------------------------------

for trajectory in range(n_train):

    for t in range(
        inputs.shape[1] - window - horizon + 1
    ):

        x_sequence = inputs[
            trajectory,
            t:t + window,
            :
        ]

        y_sequence = target[
            trajectory,
            t + window:t + window + horizon
        ]

        X_train_sequences.append(x_sequence)
        y_train_sequences.append(y_sequence)


# ------------------------------------------------------------
# 5. Create testing sequences
# ------------------------------------------------------------

for trajectory in range(n_train, n_trajectories):

    for t in range(
        inputs.shape[1] - window - horizon + 1
    ):

        x_sequence = inputs[
            trajectory,
            t:t + window,
            :
        ]

        y_sequence = target[
            trajectory,
            t + window:t + window + horizon
        ]

        X_test_sequences.append(x_sequence)
        y_test_sequences.append(y_sequence)


# Convert lists to tensors

X_train = torch.stack(X_train_sequences)
y_train = torch.stack(y_train_sequences)

X_test = torch.stack(X_test_sequences)
y_test = torch.stack(y_test_sequences)


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ------------------------------------------------------------
# 6. Normalize input
# ------------------------------------------------------------

input_mean = X_train.mean(dim=(0, 1))
input_std = X_train.std(dim=(0, 1))

X_train_norm = (
    X_train - input_mean
) / (input_std + 1e-8)

X_test_norm = (
    X_test - input_mean
) / (input_std + 1e-8)


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


# ------------------------------------------------------------
# 8. LSTM model
# ------------------------------------------------------------

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=12,
        hidden_size=64,
        num_layers=2,
        horizon=5
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            horizon
        )


    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = output[:, -1, :]

        prediction = self.fc(last_hidden)

        return prediction


# ------------------------------------------------------------
# 9. Create model
# ------------------------------------------------------------

model = LSTMModel(
    input_size=12,
    hidden_size=64,
    num_layers=2,
    horizon=5
)

print(model)


# ------------------------------------------------------------
# 10. Training setup
# ------------------------------------------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100


# ------------------------------------------------------------
# 11. Training
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
# 12. Test prediction
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    test_prediction_norm = model(
        X_test_norm.float()
    )


# ------------------------------------------------------------
# 13. Convert predictions back to original units
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
# 14. Show first predictions
# ------------------------------------------------------------

print("First 10 real values:")
print(y_test[:10])

print("First 10 predicted values:")
print(test_prediction[:10])


# ------------------------------------------------------------
# 15. Evaluation
# ------------------------------------------------------------

error = (
    test_prediction - y_test
)

mae = torch.mean(
    torch.abs(error)
)

rmse = torch.sqrt(
    torch.mean(error ** 2)
)

print("MAE:", mae.item())
print("RMSE:", rmse.item())


# ------------------------------------------------------------
# 16. Plot first test trajectory
# ------------------------------------------------------------

trajectory = 0

steps_per_trajectory = (
    inputs.shape[1]
    - window
    - horizon
    + 1
)

start = trajectory * steps_per_trajectory
end = (
    trajectory + 1
) * steps_per_trajectory


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
# 17. Save plots for all horizons
# ------------------------------------------------------------

for horizon_index in range(5):

    plt.figure(figsize=(10, 5))

    plt.plot(
        real[:, horizon_index],
        label="Real"
    )

    plt.plot(
        predicted[:, horizon_index],
        label="Predicted"
    )

    plt.xlabel("Prediction step")

    plt.ylabel("Target")

    plt.title(
        f"Experiment 13 - LSTM - Horizon {horizon_index + 1}"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    filename = (
        f"experiment13_horizon_{horizon_index + 1}.png"
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
# 18. Final result
# ------------------------------------------------------------

print("=================================")
print("EXPERIMENT 13 COMPLETED")
print("LSTM")
print("MAE:", mae.item())
print("RMSE:", rmse.item())
print("=================================")