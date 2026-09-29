import torch
import torch.nn as nn
import torch.nn.functional as F


class DownscalingCNN(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):

        super().__init__()

        # -----------------------------------------
        # Encoder
        # -----------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        # -----------------------------------------
        # Feature processing
        # -----------------------------------------

        self.features = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        # -----------------------------------------
        # Output
        # -----------------------------------------

        self.output_layer = nn.Conv2d(
            64,
            out_channels,
            kernel_size=3,
            padding=1
        )


    def forward(self, x, target_size):

        # Encoder
        x = self.encoder(x)

        # Feature extraction
        x = self.features(x)

        # Upsampling directly to target resolution
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        # Output
        x = self.output_layer(x)

        return x