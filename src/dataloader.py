import os
import sys
import zipfile

sys.path.append("src/")

from utils import config, CustomException


class Loader:
    def __init__(self, image_path=None, channels=3, image_size=256, split_size=0.20):
        self.image_path = image_path
        self.channels = channels
        self.image_size = image_size
        self.split_size = split_size

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


if __name__ == "__main__":
    loader = Loader(image_path="./data/raw/dataset1.zip")
    # loader.unzip_folder()
