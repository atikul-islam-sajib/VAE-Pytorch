import os
import sys
import torch
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

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

        self.model = self.init["model"].to(self.device)
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

        if self.weight_init:
            self.model.apply(weight_init)

        if self.lr_scheduler:
            self.scheduler = StepLR(
                optimizer=self.optimizer, step_size=self.step_size, gamma=self.gamma
            )

        self.loss = float("inf")
        self.history = {"train_loss": [], "valid_loss": []}

    def l1_regularization_loss(self, model):
        if isinstance(model, VariationalAutoEncoder):
            return self.weight_decay * sum(
                torch.norm(params, 1) for params in model.parameters()
            )

        else:
            raise CustomException(
                "Model is not an instance of VariationalAutoEncoder", sys
            )

    def l2_regularization_loss(self, model):
        if isinstance(model, VariationalAutoEncoder):
            return self.weight_decay * sum(
                torch.norm(params, 2) for params in model.parameters()
            )

        else:
            raise CustomException(
                "Model is not an instance of VariationalAutoEncoder", sys
            )

    def update_model_loss(self, **kwargs):
        self.optimizer.zero_grad()

        X = kwargs["X"]
        y = kwargs["y"]

        predicted, mean, log_variance = self.model(X)

        criterion_loss = self.criterion(predicted, y)
        kl_divergence_loss = self.kl_divergence_loss(mean, log_variance)

        self.total_loss = criterion_loss + kl_divergence_loss

        if self.l1_regularization:
            self.total_loss += self.l1_regularization_loss(self.model)

        if self.l2_regularization:
            self.total_loss += self.l2_regularization_loss(self.model)

        self.total_loss.backward()
        self.optimizer.step()

        return self.total_loss.item()

    def show_progress(self, **kwargs):
        if self.verbose:
            print(
                "Epochs:[{}/{}] - train_loss: [{:.4f}] - valid_loss:{:.4f}".format(
                    kwargs["epoch"],
                    self.epochs,
                    kwargs["train_loss"],
                    kwargs["valid_loss"],
                )
            )
        else:
            print(
                "Epochs:[{}/{}] is completed".capitalize().format(
                    kwargs["epoch"], self.epochs
                )
            )

    def save_images(self, **kwargs):
        epoch = kwargs["epoch"]

        X, y = next(iter(self.train_dataloader))
        X = X.to(self.device)
        y = y.to(self.device)

        predicted, _, _ = self.model(X)
        if epoch % 100 == 0:
            save_image(
                predicted,
                os.path.join(
                    config()["path"]["TRAIN_IMAGES_PATH"],
                    "train_image{}.png".format(epoch + 1),
                ),
            )

    def saved_checkpoints(self, **kwargs):
        epoch = kwargs["epoch"]
        train_loss = kwargs["train_loss"]
        valid_loss = kwargs["valid_loss"]

        if self.loss > valid_loss:
            self.loss = valid_loss

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                },
                os.path.join(config()["path"]["TEST_MODELS"], "best_model.pth"),
            )

        torch.save(
            self.model.state_dict(),
            os.path.join(config()["path"]["TEST_MODELS"], "model{}.pth".format(epoch)),
        )

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.train_loss = []
            self.valid_loss = []

            for _, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.train_loss.append(self.update_model_loss(X=X, y=y))

            for _, (X, y) in enumerate(self.valid_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                predicted, mean, log_variance = self.model(X)

                predicted_loss = self.criterion(predicted, y)
                kl_divergence_loss = self.kl_divergence_loss(mean, log_variance)

                total_loss = predicted_loss + kl_divergence_loss

                self.valid_loss.append(total_loss.item())

            try:
                self.show_progress(
                    epoch=epoch + 1,
                    train_loss=np.mean(self.train_loss),
                    valid_loss=np.mean(self.valid_loss),
                )
            except Exception as e:
                print("An error occured: {}".format(e))
                traceback.print_exc()

            try:
                self.save_images(epoch=epoch + 1)
            except Exception as e:
                print("An error occured: {}".format(e))
                traceback.print_exc()

            try:
                self.saved_checkpoints(
                    epoch=epoch + 1,
                    train_loss=np.mean(self.train_loss),
                    valid_loss=np.mean(self.valid_loss),
                )
            except Exception as e:
                print("An error occured: {}".format(e))
                traceback.print_exc()

            self.history["train_loss"].extend(np.mean(self.train_loss))
            self.history["valid_loss"].extend(np.mean(self.valid_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model for VAE".title())
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.002, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer".capitalize()
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.85,
        help="Gamma for learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--adam", action="store_true", help="Use Adam optimizer".capitalize()
    )
    parser.add_argument(
        "--SGD", action="store_true", help="Use SGD optimizer".capitalize()
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to use".capitalize()
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose mode".capitalize()
    )
    parser.add_argument(
        "--lr_scheduler",
        action="store_true",
        help="Use learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--weight_init",
        action="store_true",
        help="Use weight initialization".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        action="store_true",
        help="Use L1 regularization".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        action="store_true",
        help="Use L2 regularization".capitalize(),
    )
    parser.add_argument(
        "--MLFlow", action="store_true", help="Use MLFlow for tracking".capitalize()
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        adam=args.adam,
        SGD=args.SGD,
        device=args.device,
        lr_scheduler=args.lr_scheduler,
        weight_init=args.weight_init,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
        verbose=args.verbose,
        MLFlow=args.MLFlow,
    )

    trainer.train()
