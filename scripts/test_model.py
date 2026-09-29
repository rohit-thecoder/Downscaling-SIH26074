import torch

from src.models.downscaling_cnn import DownscalingCNN


print("=" * 70)
print("TESTING CNN")
print("=" * 70)


# Fake ERA5 input
x = torch.randn(
    2,
    3,
    25,
    21
)


model = DownscalingCNN(
    in_channels=3,
    out_channels=3
)


output = model(
    x,
    target_size=(61, 51)
)


print("\nInput shape:")
print(x.shape)


print("\nOutput shape:")
print(output.shape)


print("\nModel:")
print(model)