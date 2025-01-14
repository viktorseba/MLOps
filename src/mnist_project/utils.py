import matplotlib.pyplot as plt


def show_image_and_target(images, targets, show=False):
    """
    Display 25 torch arrays of size (1, 1, 28, 28) along with their corresponding targets.

    images: list of torch.Tensor with shape (1, 1, 28, 28)
    targets: list of labels (targets) corresponding to each image
    """

    # Check if the input is valid
    if len(images) != 25 or len(targets) != 25:
        raise ValueError("The function expects 25 images and 25 corresponding targets.")

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.flatten()

    for img, target, ax in zip(images, targets, axes):
        # Convert the torch tensor to a numpy array and display it
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(f"Target: {target}")
        ax.axis("off")

    plt.tight_layout()
    if show:
        plt.show()


# Example usage:
# images = [torch.rand((1, 28, 28)) for _ in range(25)]
# targets = list(range(25))
# show_images(images, targets)
