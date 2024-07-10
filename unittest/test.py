import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("src/")

from dataloader import Loader
from encoder import EncoderBlock
from decoder import DecoderBlock
from VAE import VariationalAutoEncoder
from helper import helpers
from utils import load, config


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.train_dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "train_dataloader.pkl"
            )
        )
        self.valid_dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
            )
        )

    def test_dataloader(self):
        self.assertEqual(
            self.train_dataloader.__class__, torch.utils.data.dataloader.DataLoader
        )
        self.assertEqual(
            self.valid_dataloader.__class__, torch.utils.data.dataloader.DataLoader
        )

    def test_quantity_train_dataloader(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.train_dataloader), 12)

    def test_quantity_valid_dataloader(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.valid_dataloader), 6)

    def test_encoder(self):
        in_channels = 3
        out_channels = 128

        layers = []

        for _ in range(2):
            layers.append(
                EncoderBlock(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels
            out_channels //= 2

        model = nn.Sequential(*layers)

        self.assertEqual(
            model(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 64, 64, 64])
        )

    def test_decoder(self):
        layers = []

        in_channels = 64
        out_channels = 128

        layers.append(DecoderBlock(in_channels=in_channels, out_channels=out_channels))

        in_channels = out_channels

        layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        model = nn.Sequential(*layers)

        self.assertEqual(
            model(torch.randn(1, 64, 64, 64)).size(), torch.Size([1, 3, 256, 256])
        )


if __name__ == "__main__":
    unittest.main()
