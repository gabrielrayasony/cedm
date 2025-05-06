
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig