import json
from pathlib import Path

import torch

from src.models.downscaling_cnn import DownscalingCNN


class WeatherDownscaler:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        root = Path(__file__).resolve().parents[2]

        model_path = root / "models" / "downscaling_cnn.pth"

        stats_path = (
            root
            / "data"
            / "processed"
            / "tensors"
            / "normalization.json"
        )

        self.model = DownscalingCNN(
            in_channels=3,
            out_channels=3
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        self.model.eval()

        with open(stats_path) as f:
            stats = json.load(f)

        self.X_mean = torch.tensor(
            stats["X"]["mean"],
            dtype=torch.float32
        ).view(1, 3, 1, 1).to(self.device)

        self.X_std = torch.tensor(
            stats["X"]["std"],
            dtype=torch.float32
        ).view(1, 3, 1, 1).to(self.device)

        self.Y_mean = torch.tensor(
            stats["Y"]["mean"],
            dtype=torch.float32
        ).view(1, 3, 1, 1).to(self.device)

        self.Y_std = torch.tensor(
            stats["Y"]["std"],
            dtype=torch.float32
        ).view(1, 3, 1, 1).to(self.device)


    def predict(self, x):

        """
        x shape:
        (batch, 3, 25, 21)

        Channels:
        0 = t2m
        1 = u10
        2 = v10

        Returns:
        (batch, 3, 61, 51)
        """

        x = x.to(
            self.device,
            dtype=torch.float32
        )

        # Normalize input
        x_normalized = (
            x - self.X_mean
        ) / self.X_std

        # CNN inference
        with torch.no_grad():

            prediction_normalized = self.model(
                x_normalized,
                target_size=(61, 51)
            )

        # Denormalize output
        prediction = (
            prediction_normalized * self.Y_std
            + self.Y_mean
        )

        return prediction.cpu()