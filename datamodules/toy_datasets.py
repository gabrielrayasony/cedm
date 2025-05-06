import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_swiss_roll, make_blobs, make_circles, make_moons
import lightning as L
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Union, List


class ToyDataModule(L.LightningDataModule):
    """A Lightning DataModule for 2D toy datasets.
    
    This module provides a standardized way to load and preprocess various 2D toy datasets
    for training diffusion models. It supports multiple dataset types and preprocessing options.
    
    Attributes:
        data (torch.Tensor): The raw dataset
        train (Dataset): Training dataset
        val (Dataset): Validation dataset
    """
    
    def __init__(
        self,
        dataset_name: str = "checkerboard",
        num_samples: int = 1000,
        noise: float = 0.05,
        batch_size: int = 128,
        num_workers: int = 0,
        min_max_normalize: bool = False,
        normalize: bool = False,
        rescale: bool = False,
        a: float = 1.0,
        seed: int = 42,
        train_val_split: float = 0.8
    ):
        """Initialize the ToyDataModule.
        
        Args:
            dataset_name: Name of the toy dataset to generate
            num_samples: Number of samples to generate
            noise: Noise level for the dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            min_max_normalize: Whether to apply min-max normalization
            normalize: Whether to apply standardization
            rescale: Whether to rescale the data
            a: Scaling factor for rescaling
            seed: Random seed for reproducibility
            train_val_split: Proportion of data to use for training
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.noise = noise
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_max_normalize = min_max_normalize
        self.normalize = normalize
        self.rescale = rescale
        self.a = a
        self.seed = seed
        self.train_val_split = train_val_split
        self.data = None
        self.train = None
        self.val = None
        
    def prepare_data(self) -> None:
        """Download or prepare the dataset.
        
        This method is called once per node when using distributed training.
        """
        self.data = create_toy_dataset(
            self.dataset_name,
            self.num_samples,
            self.noise
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create dataset and split into train and val when needed.
        
        Args:
            stage: Either 'fit' (training) or 'test'
        """
        if stage == "fit" or stage is None:
            self.dataset = ToyDataset(
                self.data,
                self.min_max_normalize,
                self.normalize,
                self.rescale,
                self.a
            )
            
            # Split into train and validation
            train_size = int(self.train_val_split * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            self.train, self.val = torch.utils.data.random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader.
        
        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


class ToyDataset(Dataset):
    """A PyTorch Dataset for 2D toy data with various preprocessing options.
    
    This dataset class handles the preprocessing of 2D data including normalization,
    standardization, and rescaling. It supports multiple preprocessing steps that can
    be enabled/disabled through the constructor parameters.
    
    Attributes:
        data (torch.Tensor): The preprocessed dataset
        min_max_normalize (bool): Whether to apply min-max normalization
        normalize (bool): Whether to apply standardization
        rescale (bool): Whether to rescale the data
        a (float): Scaling factor for rescaling
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        min_max_normalize: bool = False,
        normalize: bool = True,
        rescale: bool = True,
        a: float = 1.0
    ):
        """Initialize the ToyDataset.
        
        Args:
            data: Input data as either torch.Tensor or numpy array
            min_max_normalize: Whether to apply min-max normalization
            normalize: Whether to apply standardization
            rescale: Whether to rescale the data
            a: Scaling factor for rescaling
        """
        # Convert input to torch tensor if needed
        self.data = data.clone().detach().requires_grad_(False) if isinstance(data, torch.Tensor) else torch.FloatTensor(data)
        
        # Store preprocessing options
        self.min_max_normalize = min_max_normalize
        self.normalize = normalize
        self.rescale = rescale
        self.a = a
        
        # Apply preprocessing in order
        if min_max_normalize:
            self._min_max_normalize()
        if normalize:
            self._normalize()
        if rescale:
            self._rescale(a)
    
    def _normalize(self) -> None:
        """Standardize data to have zero mean and unit variance."""
        print("Standardizing data...")
        mean = self.data.mean(dim=0)
        std = self.data.std(dim=0)
        if (std == 0).any():
            raise ValueError("Cannot normalize data: some features have zero standard deviation")
        self.data = (self.data - mean) / std

    def _min_max_normalize(self) -> None:
        """Normalize data to be between -1 and 1."""
        print("Min-max normalizing data...")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data = torch.FloatTensor(scaler.fit_transform(self.data.numpy()))

    def _rescale(self, a: float) -> None:
        """Rescale data from [0, 1] to [-a, a].
        
        Args:
            a: Scaling factor
        """
        print(f"Rescaling data to range [-{a}, {a}]...")
        self.data = self.data * (2 * a) - a

    def __len__(self) -> int:
        """Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to get
            
        Returns:
            Tuple[torch.Tensor, int]: The sample and its index
        """
        return self.data[index], index


def create_toy_dataset(name, num_samples=1000, noise=0.05):
    """Factory function for toy datasets"""
    if name == 'swiss_roll':
        data, _ = make_swiss_roll(n_samples=num_samples, noise=noise)
        data = data[:, [0, 2]]
    elif name == "moons":
        data, _ = make_moons(n_samples=num_samples, noise=noise)
    elif name == 'gaussian_mixture':
        centers = np.random.uniform(-5, 5, (5, 2))
        data, _ = make_blobs(n_samples=num_samples, centers=centers)
    elif name == 'circles':
        data, _ = make_circles(n_samples=num_samples, factor=0.5, noise=noise)
    elif name == 'checkerboard':
        x = np.random.uniform(-1, 1, 2 * num_samples)
        y = np.random.uniform(-1, 1, 2 * num_samples)
        mask = ((np.floor(x * 2) + np.floor(y * 2)) % 2) == 0
        data = np.stack([x[mask], y[mask]], axis=1)
    elif name == 'spirals':
        n = np.sqrt(np.random.rand(num_samples // 2)) * 720 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.normal(0, noise, n.size)
        d1y = np.sin(n) * n + np.random.normal(0, noise, n.size)
        d2x = np.cos(n) * n + np.random.normal(0, noise, n.size)
        d2y = -np.sin(n) * n + np.random.normal(0, noise, n.size)
        data = np.vstack((np.hstack((d1x, d2x)), np.hstack((d1y, d2y)))).T 
    elif name == 'text-points':
        data = create_text_point_cloud(text="SONY", font_size=200, num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return data


def create_text_point_cloud(text, font_size, num_samples):
    """Create a point cloud from text"""
    potential_fonts = [
        "Arial.ttf",
        "arial.ttf", 
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    
    font_path = None
    for path in potential_fonts:
        if os.path.exists(path):
            font_path = path
            break
            
    if font_path is None:
        raise RuntimeError("Could not find a suitable font. Please specify font_path manually.")
    
    image_size = (1000, 400)
    image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0,0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
        draw.text(text_position, text, fill=255, font=font)
    except Exception as e:
        raise RuntimeError(f"Error rendering text with font {font_path}: {str(e)}")

    binary_image = np.array(image) > 128
    y_coords, x_coords = np.where(binary_image)
    
    if len(x_coords) == 0:
        raise RuntimeError("No pixels found in rendered text. Try increasing font size.")
        
    # Fix the coordinate system: 
    # 1. Flip y-coordinates (image coordinates are top-to-bottom)
    # 2. Rescale to [-1, 1] range in both x and y
    y_coords = image_size[1] - y_coords  # Flip y-coordinates

    # Normalize correctly: Scale to [-1, 1] range in both x and y
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_coords = (x_coords - x_min) / (x_max - x_min) * 2 - 1  # Scale x to [-1,1]
    y_coords = (y_coords - y_min) / (y_max - y_min) * 2 - 1  # Scale y to [-1,1]

    coords = np.column_stack([x_coords, y_coords])

    if len(coords) > num_samples:
        sampled_indices = np.random.choice(len(coords), size=num_samples, replace=False)
        coords = coords[sampled_indices]
    return coords


# Example usage
# toy_dm = ToyDataModule(dataset_name="checkerboard", num_samples=10000, noise=0.05, batch_size=5000, num_workers=0)
# toy_dm.prepare_data()
# toy_dm.setup()

# train_loader = toy_dm.train_dataloader()
# x, labels = next(iter(train_loader))
# print(x.shape)
# import matplotlib.pyplot as plt
# plt.plot(x[:, 0], x[:, 1], ".")
# plt.show()
