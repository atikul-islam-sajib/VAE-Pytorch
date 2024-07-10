import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt

sys.path.append("src/")

from VAE import VariationalAutoEncoder
from utils import load, config, device_init


class Tester:
    def __init__(self, model_path="best", device="cuda"):
        self.model_path = model_path
        self.device = device

        self.device = device_init(device=self.device)

    def select_the_model(self):
        if self.model_path == "best":
            best_model_path = os.path.join(
                config()["path"]["TEST_MODELS"], "best_model.pth"
            )

            return torch.load(best_model_path)["model"]
        else:
            best_model_path = self.model_path
            model_state = torch.load(best_model_path)

            return model_state

    def plot(self):
        valid_dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
            )
        )

        X, y = next(iter(valid_dataloader))
        X = X.to(self.device)
        y = y.to(self.device)

        predicted, _, _ = self.model(X)

        number_of_rows = (X.size(0) + 1) // 2
        number_of_columns = 2

        plt.figure(figsize=(15, 15))

        for index in range(X.size(0)):
            pred = predicted[index].permute(1, 2, 0).cpu().detach().numpy()
            actual = X[index].permute(1, 2, 0).cpu().detach().numpy()
            target = y[index].permute(1, 2, 0).cpu().detach().numpy()

            pred = (pred - pred.min()) / (pred.max() - pred.min())
            actual = (actual - actual.min()) / (actual.max() - actual.min())
            target = (target - target.min()) / (target.max() - target.min())

            plt.subplot(number_of_rows, number_of_columns * 3, 3 * index + 1)
            plt.imshow(actual)
            plt.axis("off")
            plt.title("Actual")

            plt.subplot(number_of_rows, number_of_columns * 3, 3 * index + 2)
            plt.imshow(pred)
            plt.axis("off")
            plt.title("Predicted")

            plt.subplot(number_of_rows, number_of_columns * 3, 3 * index + 3)
            plt.imshow(target)
            plt.axis("off")
            plt.title("Target")

        plt.tight_layout()
        plt.savefig(
            os.path.join(config()["path"]["VALID_IMAGES_PATH"], "test_result.png")
        )
        plt.show()

        print(
            "The test result is saved in the path {}".format(
                config()["path"]["VALID_IMAGES_PATH"]
            )
        )

    def test(self):
        self.model = VariationalAutoEncoder().to(self.device)
        self.model.load_state_dict(self.select_the_model())

        self.model.eval()
        self.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tester for Variational Auto Encoder".title()
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config()["tester"]["model"],
        help="Path to the model to be tested".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["tester"]["device"],
        help="Device to be used".capitalize(),
    )

    args = parser.parse_args()

    tester = Tester(model_path=args.model, device=args.device)

    tester.test()
