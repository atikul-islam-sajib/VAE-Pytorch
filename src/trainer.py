import sys
import torch

sys.path.append("src/")


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        momentum=0.9,
        weight_decay=0.0001,
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
        self.device = device
        self.verbose = verbose
        self.lr_scheduler = lr_scheduler
        self.weight_init = weight_init
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.MLFlow = MLFlow
