import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("src/")


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1

        self.decoder = self.decoder_block()

    def decoder_block(self):
        layers = [
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ]

        return nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.decoder(x)
        else:
            raise Exception("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoder Block for Variational Autoencoder".capitalize()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=128,
        help="Number of output channels".capitalize(),
    )

    args = parser.parse_args()

    layers = []

    in_channels = 64
    out_channels = 128

    layers.append(DecoderBlock(in_channels=in_channels, out_channels=out_channels))

    in_channels = out_channels

    layers.append(
        nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=3, kernel_size=4, stride=2, padding=1
        )
    )

    model = nn.Sequential(*layers)

    assert model(torch.randn(1, 64, 64, 64)).size() == (1, 3, 256, 256)
