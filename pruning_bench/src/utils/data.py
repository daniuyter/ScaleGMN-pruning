import os, random, torch, numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def _seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy, and Torch; return a torch.Generator for DataLoader use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_cifar10_data(grayscale=False, batch_size=128, val_split=0.2, device=None, num_workers=2, data_root=None):
    """
    Get CIFAR-10 datasets and dataloaders in either RGB or grayscale format.
    
    Args:
        grayscale (bool): If True, convert images to grayscale, otherwise keep RGB.
        batch_size (int): Batch size for the dataloaders.
        val_split (float): Fraction of training data to use for validation (0.0 to 1.0).
        device (torch.device): Device to use for pinning memory. If None, detect automatically.
        num_workers (int): Number of subprocesses to use for data loading.
        data_root (str): Root directory where the dataset is stored or will be downloaded.
    
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' dataloaders, and 'train_dataset', 
              'val_dataset', and 'test_dataset' datasets.
    """
    if data_root is None:
        # Assumes this script is in src/utils/ and data should be in project_root/data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) 
        data_root = os.path.join(project_root, 'data')
        os.makedirs(data_root, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations based on grayscale parameter
    if grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)  # RGB CIFAR-10 stats
        ])
    
    # Load datasets
    train_full_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create validation split if requested
    if val_split > 0:
        val_size = int(len(train_full_dataset) * val_split)
        train_size = len(train_full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
    else:
        train_dataset = train_full_dataset
        val_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')  # Pin memory only if using GPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    result = {
        'train': train_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }
    
    # Add validation data if created
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
        result['val'] = val_loader
        result['val_dataset'] = val_dataset
    
    # Print dataset information
    print(f"Dataset type: {'Grayscale' if grayscale else 'RGB'} CIFAR-10")
    print(f"Training set: {len(train_dataset)} images")
    if val_dataset is not None:
        print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    return result