import sys
import torch
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from helper import helpers
from VAE import VariationalAutoEncoder
from mse import MSELoss
from kl_divergence import KLDivergence
from utils import dump, load, config, weight_init, device_init, CustomException


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        momentum=0.9,
        weight_decay=0.0001,
        step_size=10,
        gamma=0.85,
        adam=True,
        SGD=False,
        device="cuda",
        verbose=True,
        lr_scheduler=False,
        weight_init=False,
        l1_regularization=False,
        l2_regularization=False,
        MLFlow=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.verbose = verbose
        self.lr_scheduler = lr_scheduler
        self.weight_init = weight_init
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.MLFlow = MLFlow

        self.init = helpers(
            adam=self.adam,
            SGD=self.SGD,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
        )

        try:
            self.device = device_init(device=self.device)
        except Exception as e:
            raise CustomException(e, sys)

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.model = self.init["model"]
        self.optimizer = self.init["optimizer"]

        self.criterion = self.init["criterion"]
        self.kl_divergence_loss = self.init["kl_diversance_loss"]

        assert (
            self.init["train_dataloader"].__class__
            == torch.utils.data.dataloader.DataLoader
        )
        assert (
            self.init["valid_dataloader"].__class__
            == torch.utils.data.dataloader.DataLoader
        )

        assert self.init["model"].__class__ == VariationalAutoEncoder
        assert self.init["optimizer"].__class__ == torch.optim.Adam

        assert self.init["criterion"].__class__ == MSELoss
        assert self.init["kl_diversance_loss"].__class__ == KLDivergence

        self.model = self.model.to(self.device)

        if self.weight_init:
            self.model.apply(weight_init)

        if self.lr_scheduler:
            self.scheduler = StepLR(
                optimizer=self.optimizer, step_size=self.step_size, gamma=self.gamma
            )

        self.loss = float("inf")
        self.history = {"train_loss": [], "valid_loss": []}


if __name__ == "__main__":
    trainer = Trainer(epochs=1, device="mps")
