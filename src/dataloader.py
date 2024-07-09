import os
import cv2
from PIL import Image
from tqdm import tqdm
import sys
import zipfile
from torchvision import transforms

sys.path.append("src/")

from utils import config, CustomException


class Loader:
    def __init__(self, image_path=None, channels=3, image_size=256, split_size=0.20):
        self.image_path = image_path
        self.channels = channels
        self.image_size = image_size
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


if __name__ == "__main__":
    loader = Loader(image_path="./data/raw/dataset1.zip")
    # loader.unzip_folder()
    loader.extract_features()
