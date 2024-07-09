import os
import sys
import torch
import traceback
import torch.optim as optim

sys.path.append("src/")

from mse import MSELoss
from VAE import VariationalAutoEncoder
from kl_divergence import KLDivergence
from utils import load, config, CustomException


def load_dataloader():
    processed_datapath = config()["path"]["PROCESSED_DATA_PATH"]

    if os.path.exists(processed_datapath):

        train_dataloader = load(
            filename=os.path.join(processed_datapath, "train_dataloader.pkl")
        )
        valid_dataloader = load(
            filename=os.path.join(processed_datapath, "valid_dataloader.pkl")
        )

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }

    else:
        raise CustomException("Processed data not found".capitalize())


def helpers(**kwargs):
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]

    channels = config()["VAE"]["channels"]
    image_size = config()["VAE"]["image_size"]

    model = VariationalAutoEncoder(channels=channels, image_size=image_size)

    if adam:
        optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(beta1, beta2))

    if SGD:
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

    criterion = MSELoss(reduction="mean")
    kl_diversance_loss = KLDivergence()

    try:
        dataloader = load_dataloader()

    except CustomException as e:
        print("An eeror is occured: ", e)
        traceback.print_exc()

    except Exception as e:
        print("An error is occured: ", e)
        traceback.print_exc()

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "kl_diversance_loss": kl_diversance_loss,
    }


if __name__ == "__main__":
    init = helpers(adam=True, SGD=False, lr=0.001, beta1=0.9, beta2=0.999, momentum=0.9)

    assert init["train_dataloader"].__class__ == torch.utils.data.dataloader.DataLoader
    assert init["valid_dataloader"].__class__ == torch.utils.data.dataloader.DataLoader

    assert init["model"].__class__ == VariationalAutoEncoder
    assert init["optimizer"].__class__ == torch.optim.Adam

    assert init["criterion"].__class__ == MSELoss
    assert init["kl_diversance_loss"].__class__ == KLDivergence
