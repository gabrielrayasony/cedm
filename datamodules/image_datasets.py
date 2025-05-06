import torch 
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST
import lightning as L
import matplotlib.pyplot as plt


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=0, data_dir='/mnt/ssd1/datasets/', pin_memory=True):
        super().__init__()
        self.datadir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(self.datadir, train=True, download=True)
        CIFAR10(self.datadir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = CIFAR10(self.datadir, train=True, download=False, transform=self.transform)
            self.val = CIFAR10(self.datadir, train=False, download=False, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
        
    def visualize_batch(self, batch_size=25, save_path=None):
        """Visualize a batch of CIFAR10 images in a grid layout using torchvision's make_grid.
        
        Args:
            batch_size (int): Number of images to display
            save_path (str, optional): Path to save the visualization
        """
        dataloader = self.train_dataloader()
        images, _ = next(iter(dataloader))
        images = images[:batch_size]
        print(images.min(), images.max())
        # Create grid of images
        grid = torchvision.utils.make_grid(images, nrow=int(batch_size**0.5), normalize=True)
        
        # Convert to numpy and show
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


class BinaryMNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = '/mnt/ssd1/datasets/',  # Updated to match CIFAR10's style
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        threshold: float = 0.5
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.threshold = threshold
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.binarize
        ])
        
        self.dims = (1, 28, 28)
        self.num_classes = 1
        
    def binarize(self, x):
        return (x > self.threshold).float()
    
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(
                self.data_dir,
                train=True,
                transform=self.transform
            )
            
            train_size = int(0.95 * len(mnist_full))
            val_size = len(mnist_full) - train_size
            
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def visualize_batch(self, batch_size=25, save_path=None):
        dataloader = self.train_dataloader()
        images, _ = next(iter(dataloader))
        images = images[:batch_size]
        
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        axes = axes.ravel()
        
        for idx, img in enumerate(images):
            axes[idx].imshow(img.squeeze(), cmap='binary')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()