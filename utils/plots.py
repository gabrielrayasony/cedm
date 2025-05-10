
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_utils import rescaling_inv
sns.set_theme()


 
#-----------------------------------------------------------------------------------------------------------------
# 2D plotting functions
#-----------------------------------------------------------------------------------------------------------------  

def plot_2d_data(x, 
            workdir=None, 
            name="plot", 
            figsize=(6, 6), 
            xlim=None, 
            ylim=None, 
            alpha=0.5, 
            size=10, 
            color=plt.cm.cividis(0.2), 
            show=False, 
            dpi=200):
    """
    Simple 2D plotting function.

    Args:
        x (array-like): 2D data to plot (shape: [n_samples, 2]).
        workdir (str, optional): Directory to save the plot. If None, the plot won't be saved.
        name (str): Name of the saved plot file (without extension). Defaults to "plot".
        figsize (tuple): Figure size as (width, height). Defaults to (6, 6).
        xlim (tuple or float, optional): x-axis limits. If float, uses (-xlim, xlim).
        ylim (tuple or float, optional): y-axis limits. If float, uses (-ylim, ylim).
        alpha (float): Transparency of points. Defaults to 0.5.
        color (str or array-like, optional): Color for points. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    # Create the plot
    fig = plt.figure(figsize=figsize)
    plt.scatter(x[:, 0], x[:, 1], alpha=alpha, c=[color], s=size,)

    # Set axis limits
    if isinstance(xlim, (tuple, list)):
        plt.xlim(xlim)
    elif isinstance(xlim, (int, float)):
        plt.xlim(-xlim, xlim)

    if isinstance(ylim, (tuple, list)):
        plt.ylim(ylim)
    elif isinstance(ylim, (int, float)):
        plt.ylim(-ylim, ylim)

    # Save or display the plot
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        filepath = os.path.join(workdir, f"{name}.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved at {filepath}")
    if show or workdir is None:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_image_grid(images, nrow=8, padding=2, normalize=False, title=None, figsize=(10,10),
                    workdir=None, name="plot", dpi=200, show=False):
    """Plot a grid of images using torchvision's make_grid.
    
    Args:
        images: torch tensor of shape (N, C, H, W)
        nrow: number of images per row
        padding: padding between images
        normalize: normalize image values to [0,1]
        title: title for the plot
        figsize: figure size as (width, height)
    """
    # Convert to torch tensor if numpy array
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # Add channel dim if missing (for grayscale)
    if images.dim() == 3:
        images = images.unsqueeze(1)
        
    # Make grid
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    
    # Convert to numpy and transpose
    grid = grid.cpu().numpy()
    grid = np.transpose(grid, (1, 2, 0))
    
    # Plot
    fig = plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        filepath = os.path.join(workdir, f"{name}.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved at {filepath}")
    if show or workdir is None:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_data(x, name="training_data", figsize=(3, 3), workdir=None):
    if x.ndim == 2:
        fig = plot_2d_data(x, name=name, figsize=figsize, workdir=workdir)
    else:
        fig = plot_image_grid(rescaling_inv(x[:64]), name=name, figsize=figsize, normalize=True, workdir=workdir)
    return fig

