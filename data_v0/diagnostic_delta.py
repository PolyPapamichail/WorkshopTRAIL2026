import torch

data = torch.load(
    r".\generated_data_ramp_10h\combo_0000.pt",
    map_location="cpu",
    weights_only=False
)

Y = data["Y"]
target = Y[:, :, 0].double()

# ΔY(t) = Y(t) - Y(t-1)
delta = target[:, 1:] - target[:, :-1]

print()
print("=================================")
print("TARGET DELTA DIAGNOSTIC")
print("=================================")

print(f"Delta mean: {delta.mean().item():.6f}")
print(f"Delta std : {delta.std().item():.6f}")
print(f"Delta min : {delta.min().item():.6f}")
print(f"Delta max : {delta.max().item():.6f}")

print()
print("First 10 trajectories - first 20 deltas:")

for i in range(10):
    print()
    print(f"Trajectory {i}:")
    print(delta[i, :20].tolist())

print()
print("=================================")
print("DELTA COMPLETED")
print("=================================")