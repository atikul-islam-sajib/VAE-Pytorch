import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("src/")


class EncoderBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1

        self.encoder = self.encoder_block()

    def encoder_block(self):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder(x)

        else:
            raise Exception("Input must be a tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder Block for Variational Autoencoder".capitalize()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=128,
        help="Number of output channels".capitalize(),
    )

    args = parser.parse_args()

    in_channels = 3
    out_channels = 128

    layers = []

    for _ in range(2):
        layers.append(EncoderBlock(in_channels=in_channels, out_channels=out_channels))
        in_channels = out_channels
        out_channels //= 2

    model = nn.Sequential(*layers)

    assert model(torch.randn(1, 3, 256, 256)).size() == (1, 64, 64, 64)
