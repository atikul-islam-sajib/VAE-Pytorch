import os
import cv2
from PIL import Image
from tqdm import tqdm
import sys
import traceback
import zipfile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("src/")

from utils import config, CustomException


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

        self.train_dataloader = DataLoader(
            dataset=zip(list(self.dataset["X_train"]), list(self.dataset["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.valid_dataloader = DataLoader(
            dataset=zip(list(self.dataset["X_test"]), list(self.dataset["y_test"])),
            batch_size=self.batch_size * self.batch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    loader = Loader(image_path="./data/raw/dataset1.zip")
    # loader.unzip_folder()
