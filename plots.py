import os
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def visualize_batch(batch_tensor, nrow=8, title='', figsize=None):
    # Calculate the number of rows needed based on the batch size and desired number of columns (nrow)
    batch_size = batch_tensor.size(0)
    nrows = math.ceil(batch_size / nrow)

    # Automatically adjust the figsize based on the number of images if not provided
    if figsize is None:
        figsize = (nrow, nrows)

    # Create a grid of images
    grid_img = make_grid(batch_tensor, nrow=nrow)

    # Plotting
    plt.figure(figsize=figsize)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_image(batch_images, workdir, n=64, padding=2, pos="horizontal", nrow=3, w=5.5, file_format="jpeg", name="data_samples", scale=4, show=False):
    """
    Plot a grid of images for a given size.

    Args:
        batch_images (tensor): Tensor of size NxCxHxW.
        workdir (str): Path where the image grid will be saved.
        n (int): Number of images to display.
        padding (int): Padding size of the grid.
        pos (str): Position of the grid. Options: "horizontal", "square", or "vertical".
        nrow (int): Number of rows in the grid (only applicable when pos="vertical").
        w (float): Width size.
        file_format (str): File format for saving the image (e.g., "png").
        name (str): Name of the saved image file.
        scale (int): Scaling factor for the saved image.
        show (bool): Whether to display the image using plt.show().

    """
    if pos == "horizontal":
        sample_grid = make_grid(batch_images[:n], nrow=n, padding=padding)
    elif pos == "square":
        n = batch_images.shape[0] if batch_images.shape[0] < n else n
        sample_grid = make_grid(batch_images[:n], nrow=int(np.sqrt(n)), padding=padding)
    elif pos == "vertical":
        sample_grid = make_grid(batch_images[:n], nrow=nrow, padding=padding)
    else:
        raise ValueError("Invalid 'pos' value. Use 'horizontal', 'square', or 'vertical'.")

    fig = plt.figure(figsize=(n * w / scale, w))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu())
    fig.savefig(os.path.join(workdir, "{}.{}".format(name, file_format)), bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
