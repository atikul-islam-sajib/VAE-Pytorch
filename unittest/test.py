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


if __name__ == "__main__":
    unittest.main()
