import torch

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

target = Y[:, :, 2]

print("Target shape:", target.shape)

n_trajectories = target.shape[0]

n_train = int(0.70 * n_trajectories)

train_target = target[:n_train]
test_target = target[n_train:]

print("Total trajectories:", n_trajectories)
print("Training trajectories:", train_target.shape)
print("Testing trajectories:", test_target.shape)

# Inputs: X(t) + U(t)
inputs = torch.cat([X, U], dim=2)

# Predict the next value of the selected Y variable
features_train = inputs[:n_train, :-1, :]
target_train = target[:n_train, 1:]

features_test = inputs[n_train:, :-1, :]
target_test = target[n_train:, 1:]

print("Training features:", features_train.shape)
print("Training target:", target_train.shape)
print("Testing features:", features_test.shape)
print("Testing target:", target_test.shape)

# Convert trajectories into individual training examples
X_train = features_train.reshape(-1, 12)
y_train = target_train.reshape(-1, 1)

X_test = features_test.reshape(-1, 12)
y_test = target_test.reshape(-1, 1)

print("Final X_train:", X_train.shape)
print("Final y_train:", y_train.shape)
print("Final X_test:", X_test.shape)
print("Final y_test:", y_test.shape)

# Normalize inputs using training data only

mean = X_train.mean(dim=0)
std = X_train.std(dim=0)

X_train_norm = (X_train - mean) / (std + 1e-8)
X_test_norm = (X_test - mean) / (std + 1e-8)

# Normalize target
target_mean = y_train.mean()
target_std = y_train.std()

y_train_norm = (y_train - target_mean) / (target_std + 1e-8)
y_test_norm = (y_test - target_mean) / (target_std + 1e-8)

print("Normalization completed.")
print("Normalized X_train mean:", X_train_norm.mean().item())
print("Normalized X_train std:", X_train_norm.std().item())

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(12, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

print(model)

# Training setup
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):

    # Prediction
    prediction = model(X_train_norm.float())

    # Calculate error
    loss = loss_function(prediction, y_train_norm.float())

    # Update the neural network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

        # Prediction on the test set

model.eval()

with torch.no_grad():
    test_prediction_norm = model(X_test_norm.float())

# Convert prediction back to original units
test_prediction = (
    test_prediction_norm * (target_std + 1e-8)
    + target_mean
)

print("Test prediction shape:", test_prediction.shape)
print("Real test values shape:", y_test.shape)

print("First 10 real values:")
print(y_test[:10].flatten())

print("First 10 predicted values:")
print(test_prediction[:10].flatten())

# Evaluation metrics

error = test_prediction - y_test

mae = torch.mean(torch.abs(error))
rmse = torch.sqrt(torch.mean(error ** 2))

print("MAE:", mae.item())
print("RMSE:", rmse.item())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(y_test[:100].numpy(), label="Real")
plt.plot(prediction[:100].detach().numpy(), label="Predicted")

plt.xlabel("Time step")
plt.ylabel("Target")
plt.title("Real vs Predicted - First Test Trajectory")
plt.legend()
plt.grid()

plt.show()

import matplotlib
matplotlib.use("Agg")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Select first test trajectory
trajectory = 0

start = trajectory * 100
end = (trajectory + 1) * 100

real = y_test[start:end].flatten().detach().numpy()
predicted = test_prediction[start:end].flatten().detach().numpy()

plt.figure(figsize=(10, 5))

plt.plot(real, label="Real")
plt.plot(predicted, label="Predicted")

plt.xlabel("Time step")
plt.ylabel("Target")
plt.title("Real vs Predicted - Test Trajectory 0")

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("prediction_plot.png", dpi=150)

plt.close()

print("=================================")
print("PLOT SAVED:")
print("prediction_plot.png")
print("=================================")