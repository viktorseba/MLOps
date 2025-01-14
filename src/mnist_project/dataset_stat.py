import matplotlib.pyplot as plt
import torch
import typer
from dataset import MnistDataset
from utils import show_image_and_target


def dataset_statistics(datadir: str = "data/raw") -> None:
    """Compute dataset statistics."""
    train_dataset = MnistDataset(data_folder=datadir, train=True)
    test_dataset = MnistDataset(data_folder=datadir, train=False)
    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    show_image_and_target(train_dataset.images[:25], train_dataset.target[:25], show=False)
    plt.savefig("reports/figures/mnist_images.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
