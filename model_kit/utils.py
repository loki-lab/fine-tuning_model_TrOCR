import zipfile
from urllib.request import urlretrieve
import os
import matplotlib.pyplot as plt
from model_kit.config import ModelConfig
from transformers import TrOCRProcessor
import torch


def processor():
    p = TrOCRProcessor.from_pretrained(ModelConfig.model_name)
    return p


def download_dataset(url, save_path):
    print("Downloading and unzipping dataset...", end="")

    urlretrieve(url, save_path)

    try:
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)


def visualize(dataset_path):
    plt.figure(figsize=(15, 3))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        all_images = os.listdir(f"{dataset_path}/scut_train")
        image = plt.imread(f'{dataset_path}/scut_train/{all_images[i]}')
        plt.imshow(image)
        plt.axis("off")
        plt.title(all_images[i].split('.')[0])
    plt.show()


def check_cuda_available():
    if torch.cuda.is_available():
        print("CUDA available.")
        return "cuda"
    else:
        print("CUDA not available.")
        return "cpu"


if __name__ == "__main__":
    processor = processor()
    print(processor)
