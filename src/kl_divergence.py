import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("src/")


class KLDivergence(nn.Module):
    def __init__(self, name="KLDiversance"):
        super(KLDivergence, self).__init__()

        self.name = name

    def forward(self, mean, log_variance):
        if isinstance(mean, torch.Tensor) and isinstance(log_variance, torch.Tensor):
            return -0.5 * torch.sum(
                1 + log_variance - mean**2 - torch.exp(log_variance)
            )

        else:
            raise Exception("mean and log_variance must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLDiversance".title())
    parser.add_argument("--kl", action="store_true", help="KLDiversance".capitalize())

    mean = torch.randn(1, 64, 64, 64)
    log_variance = torch.randn(1, 64, 64, 64)

    loss = KLDivergence()

    assert type(loss(mean, log_variance)) == torch.Tensor
