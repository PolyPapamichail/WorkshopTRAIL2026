import torch

data = torch.load(
    r".\generated_data_ramp_10h\combo_0000.pt",
    map_location="cpu",
    weights_only=False
)

Y = data["Y"]
target = Y[:, :, 0].double()

delta = target[:, 1:] - target[:, :-1]

abs_delta = delta.abs()

print()
print("=================================")
print("JUMP FREQUENCY DIAGNOSTIC")
print("=================================")

for threshold in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:

    count = (abs_delta > threshold).sum().item()
    total = abs_delta.numel()
    percentage = 100.0 * count / total

    print(
        f"|Delta| > {threshold:5.2f}: "
        f"{count:6d} / {total} "
        f"({percentage:.4f}%)"
    )

print()
print("Delta statistics:")
print(f"Mean      : {delta.mean().item():.6f}")
print(f"Std       : {delta.std().item():.6f}")
print(f"Mean |Δ|  : {abs_delta.mean().item():.6f}")
print(f"Median |Δ|: {abs_delta.median().item():.6f}")
print(f"Max |Δ|   : {abs_delta.max().item():.6f}")

print()
print("=================================")
print("JUMP DIAGNOSTIC COMPLETED")
print("=================================")