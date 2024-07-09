import os
import cv2
import sys
import zipfile
import argparse
import traceback
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("src/")

from utils import config, dump, load, CustomException


class Loader:
    def __init__(
        self, image_path=None, channels=3, image_size=256, batch_size=4, split_size=0.20
    ):
        self.image_path = image_path
        self.channels = channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.actual = []
        self.target = []

    def unzip_folder(self):
        self.raw_data_path = config()["path"]["RAW_DATA_PATH"]

        if os.path.exists(self.raw_data_path):
            with zipfile.ZipFile(self.image_path, "r") as zip_file:
                zip_file.extractall(path=os.path.join(self.raw_data_path))

            print(
                "Unzip is done successfully and stoed in the path {}".format(
                    os.path.join(self.raw_data_path, "dataset")
                )
            )

        else:
            raise CustomException("Raw data path does not exist".capitalize())

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def split_dataset(self, X, y):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        else:
            raise CustomException("X and y should be list".capitalize())

    def extract_features(self):
        self.directory = os.path.join(config()["path"]["RAW_DATA_PATH"], "dataset")
        self.X = os.path.join(config()["path"]["RAW_DATA_PATH"], "dataset", "X")
        self.y = os.path.join(config()["path"]["RAW_DATA_PATH"], "dataset", "y")

        for image in tqdm(os.listdir(self.X)):
            if (image is not None) and (image in os.listdir(self.y)):
                self.imageX = os.path.join(self.X, image)
                self.imagey = os.path.join(self.y, image)

                self.imageX = cv2.imread(filename=self.imageX, flags=cv2.IMREAD_COLOR)
                self.imagey = cv2.imread(filename=self.imagey, flags=cv2.IMREAD_COLOR)

                self.imageX = cv2.cvtColor(self.imageX, cv2.COLOR_BGR2RGB)
                self.imagey = cv2.cvtColor(self.imagey, cv2.COLOR_BGR2RGB)

                self.imageX = Image.fromarray(self.imageX)
                self.imagey = Image.fromarray(self.imagey)

                self.imageX = self.transforms()(self.imageX)
                self.imagey = self.transforms()(self.imagey)

                self.actual.append(self.imageX)
                self.target.append(self.imagey)

        assert len(self.actual) == len(self.target)

        try:
            dataset = self.split_dataset(X=self.actual, y=self.target)

        except CustomException as e:
            print("An error occured: ", e)
            traceback.print_exc()

        except Exception as e:
            print("An error occured: ", e)
            traceback.print_exc()

        else:
            print("Feature extracted successfully".capitalize())

        return dataset

    def create_dataloader(self):
        self.dataset = self.extract_features()
        self.processed_data_path = config()["path"]["PROCESSED_DATA_PATH"]

        self.train_dataloader = DataLoader(
            dataset=list(zip(self.dataset["X_train"], list(self.dataset["y_train"]))),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.valid_dataloader = DataLoader(
            dataset=list(zip(self.dataset["X_test"], list(self.dataset["y_test"]))),
            batch_size=self.batch_size * self.batch_size,
            shuffle=True,
        )

        for value, filename in [
            (self.train_dataloader, "train_dataloader.pkl"),
            (self.valid_dataloader, "valid_dataloader.pkl"),
        ]:
            dump(value=value, filename=os.path.join(self.processed_data_path, filename))

        print(
            "DataLoader created successfully and stored in the path {}".capitalize().format(
                self.processed_data_path
            )
        )

    @staticmethod
    def plot_images():
        processed_data_path = config()["path"]["PROCESSED_DATA_PATH"]

        valid_dataloader = load(
            filename=os.path.join(processed_data_path, "valid_dataloader.pkl")
        )

        X, y = next(iter(valid_dataloader))

        number_of_rows = X.size(0) // 2
        number_of_columns = X.size(0) // number_of_rows

        plt.figure(figsize=(20, 10))

        for index, image in enumerate(X):
            imageX = image.permute(1, 2, 0).detach().numpy()
            imagey = y[index].permute(1, 2, 0).detach().numpy()

            imageX = (imageX - imageX.min()) / (imageX.max() - imageX.min())
            imagey = (imagey - imagey.min()) / (imagey.max() - imagey.min())

            plt.subplot(2 * number_of_rows, 2 * number_of_columns, 2 * index + 1)
            plt.title("actual".capitalize())
            plt.imshow(imageX)
            plt.axis("off")

            plt.subplot(2 * number_of_rows, 2 * number_of_columns, 2 * index + 2)
            plt.title("target".capitalize())
            plt.imshow(imagey)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["FILES_PATH"], "images.png"))
        plt.show()

        print(
            "Images saved in the path {}".format(
                config()["path"]["FILES_PATH"]
            ).capitalize()
        )


if __name__ == "__main__":
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
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Image size".capitalize(),
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

    args = parser.parse_args()

    if args.image_path:

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
