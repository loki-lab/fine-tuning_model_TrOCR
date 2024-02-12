from model_kit.utils import download_dataset
from model_kit.utils import visualize
from model_kit.config import DatasetConfig
import os


if __name__ == '__main__':
    URL = r"https://www.dropbox.com/scl/fi/vyvr7jbdvu8o174mbqgde/scut_data.zip?rlkey=fs8axkpxunwu6if9a2su71kxs&dl=1"
    asset_zip_path = os.path.join(os.getcwd(), "datasets/scut_data.zip")

    if not os.path.exists(asset_zip_path):
        download_dataset(URL, asset_zip_path)

    visualize(DatasetConfig.data_root)
