import zipfile


class Loader:
    def __init__(self, image_path=None, channels=3, image_size=256, split_size=0.20):
        self.image_path = image_path
        self.channels = channels
        self.image_size = image_size
        self.split_size = split_size

    def unzip_folder(self):
        pass
