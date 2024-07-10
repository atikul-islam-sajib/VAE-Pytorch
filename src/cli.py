import os
import sys
import argparse

sys.path.append("src/")

from utils import config
from dataloader import Loader
from VAE import VariationalAutoEncoder
from trainer import Trainer
from tester import Tester


def cli():
    parser = argparse.ArgumentParser(
        description="Dataloader for Varitional Autoencoder".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=config()["dataloader"]["image_path"],
        help="Path to the image dataset".capitalize(),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Number of channels in the image".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Batch size".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Split size".capitalize(),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Number of epochs".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Learning rate".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Beta1 for Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Beta2 for Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["trainer"]["momentum"],
        help="Momentum for SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config()["trainer"]["weight_decay"],
        help="Weight decay for SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["trainer"]["step_size"],
        help="Step size for learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["trainer"]["gamma"],
        help="Gamma for learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Use Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="Use SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Device to use".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["trainer"]["verbose"],
        help="Verbose mode".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["trainer"]["lr_scheduler"],
        help="Use learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--weight_init",
        type=bool,
        default=config()["trainer"]["weight_init"],
        help="Use weight initialization".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["trainer"]["l1_regularization"],
        help="Use L1 regularization".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["trainer"]["l2_regularization"],
        help="Use L2 regularization".capitalize(),
    )
    parser.add_argument(
        "--MLFlow",
        type=bool,
        default=config()["trainer"]["MLFlow"],
        help="Use MLFlow for tracking".capitalize(),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config()["tester"]["model"],
        help="Path to the model to be tested".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:

        loader = Loader(
            image_path=args.image_path,
            channels=args.channels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        # loader.unzip_folder()
        loader.create_dataloader()

        Loader.plot_images()
        Loader.details_dataset()

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

    elif args.test:

        tester = Tester(model_path=args.model, device=args.device)

        tester.test()


if __name__ == "__main__":
    cli()
