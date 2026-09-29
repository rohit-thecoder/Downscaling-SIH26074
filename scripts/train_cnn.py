import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.downscaling_cnn import DownscalingCNN
from src.models.losses import masked_mse_loss


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.001

INPUT_SIZE = (25, 21)
TARGET_SIZE = (61, 51)

MODEL_PATH = "models/downscaling_cnn.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CNN TRAINING")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# LOAD DATA
# ============================================================

X_train = torch.load(
    "data/processed/tensors/X_train.pt",
    weights_only=False
)

Y_train = torch.load(
    "data/processed/tensors/Y_train.pt",
    weights_only=False
)

X_val = torch.load(
    "data/processed/tensors/X_val.pt",
    weights_only=False
)

Y_val = torch.load(
    "data/processed/tensors/Y_val.pt",
    weights_only=False
)


print("\nTraining data:")
print("X:", X_train.shape)
print("Y:", Y_train.shape)

print("\nValidation data:")
print("X:", X_val.shape)
print("Y:", Y_val.shape)


# ============================================================
# DATASET
# ============================================================

train_dataset = TensorDataset(
    X_train,
    Y_train
)

val_dataset = TensorDataset(
    X_val,
    Y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = DownscalingCNN(
    in_channels=3,
    out_channels=3
).to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        prediction = model(
            X_batch,
            target_size=TARGET_SIZE
        )

        loss = masked_mse_loss(
            prediction,
            Y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            prediction = model(
                X_batch,
                target_size=TARGET_SIZE
            )

            loss = masked_mse_loss(
                prediction,
                Y_batch
            )

            val_loss += loss.item()


    val_loss /= len(val_loader)


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nModel saved to:")
print(MODEL_PATH)