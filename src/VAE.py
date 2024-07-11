import os
import sys
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import config
from encoder import EncoderBlock
from decoder import DecoderBlock


class VariationalAutoEncoder(nn.Module):
    def __init__(self, channels=3, image_size=256):
        super(VariationalAutoEncoder, self).__init__()

        self.in_channels = channels
        self.out_channels = image_size // 2

        self.kernel_size = 3
        self.stride_size = 1
        self.padding_size = 1

        self.encoder_layers = []
        self.decoder_layers = []

        for _ in range(2):
            self.encoder_layers.append(
                EncoderBlock(
                    in_channels=self.in_channels, out_channels=self.out_channels
                )
            )
            self.in_channels = self.out_channels
            self.out_channels //= 2

        self.encoder = nn.Sequential(*self.encoder_layers)

        self.mean = nn.Sequential(
            nn.Conv2d(
                in_channels=image_size // 4,
                out_channels=image_size // 4,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=False,
            )
        )

        self.log_variance = nn.Sequential(
            nn.Conv2d(
                in_channels=image_size // 4,
                out_channels=image_size // 4,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=False,
            )
        )

        self.out_channels = self.in_channels * 2

        self.decoder_layers.append(
            DecoderBlock(in_channels=self.in_channels, out_channels=self.out_channels)
        )
        self.decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.out_channels,
                    out_channels=channels,
                    kernel_size=self.kernel_size + 1,
                    stride=self.stride_size + 1,
                    padding=self.padding_size,
                ),
                nn.Tanh(),   # You can use "Sigmoid() when use BCELoss() instead of MSELoss()" 
            )
        )

        self.decoder = nn.Sequential(*self.decoder_layers)

    def reparameterization_trick(self, mean, log_variance):
        if isinstance(mean, torch.Tensor) and isinstance(log_variance, torch.Tensor):
            standard_deviation = torch.exp(0.5 * log_variance)
            eps = torch.randn_like(standard_deviation)

            z = mean + eps * standard_deviation

            return z

        else:
            raise Exception("Input must be a tensor".capitalize())

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            encoder = self.encoder(x)

            mean = self.mean(encoder)
            log_variance = self.log_variance(encoder)

            try:
                z = self.reparameterization_trick(mean=mean, log_variance=log_variance)

            except Exception as e:
                print("An error occurred: {}".format(e))

            decoder = self.decoder(z)

            return decoder, mean, log_variance

        else:
            raise Exception("Input must be a tensor".capitalize())

    @staticmethod
    def total_params(model):
        if isinstance(model, VariationalAutoEncoder):
            return sum(params.numel() for params in model.parameters())

        else:
            raise Exception("Input must be a VariationalAutoEncoder".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model for Variational Autoencoder".title()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["VAE"]["channels"],
        help="Number of channels in the input image".title(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["VAE"]["image_size"],
        help="Size of the input image".title(),
    )

    args = parser.parse_args()

    variational_autoencoder = VariationalAutoEncoder(
        channels=args.channels, image_size=args.image_size
    )

    predicted, _, _ = variational_autoencoder(torch.randn(1, 3, 256, 256))

    assert predicted.size() == (
        1,
        args.channels,
        args.image_size,
        args.image_size,
    )

    assert VariationalAutoEncoder.total_params(variational_autoencoder) == 348547

    print(summary(model=variational_autoencoder, input_size=(3, 256, 256)))

    draw_graph(
        model=variational_autoencoder,
        input_data=torch.randn(1, args.channels, args.image_size, args.image_size),
    ).visual_graph.render(
        filename=os.path.join(config()["path"]["FILES_PATH"], "VAE"), format="png"
    )

    print(
        "Model Architecture saved as VAE.png in the path {}".format(
            config()["path"]["FILES_PATH"]
        )
    )
