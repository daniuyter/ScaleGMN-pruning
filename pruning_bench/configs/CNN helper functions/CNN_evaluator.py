import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import os

layout_df_path = os.path.join(os.path.dirname(__file__), 'layout.csv')

try:
    layout_df = pd.read_csv(layout_df_path)
except FileNotFoundError:
    print("Error: layout.csv not found. Please ensure it's in the same directory.")
    layout_df = pd.DataFrame()

# Mapping TF variable names to PyTorch parameter names and their original TF shapes.
# This also implicitly defines the type of parameter (conv kernel, dense kernel, bias)
# which is used for reshaping logic. 
param_mapping_and_shape: Dict[str, Dict[str, Any]] = {
    'sequential/conv2d/bias:0': {'pytorch_name': 'conv1.bias', 'tf_shape': (16,)},
    'sequential/conv2d/kernel:0': {'pytorch_name': 'conv1.weight', 'tf_shape': (3, 3, 1, 16)}, # TF shape (H, W, In, Out)
    'sequential/conv2d_1/bias:0': {'pytorch_name': 'conv2.bias', 'tf_shape': (16,)},
    'sequential/conv2d_1/kernel:0': {'pytorch_name': 'conv2.weight', 'tf_shape': (3, 3, 16, 16)}, # TF shape (H, W, In, Out)
    'sequential/conv2d_2/bias:0': {'pytorch_name': 'conv3.bias', 'tf_shape': (16,)},
    'sequential/conv2d_2/kernel:0': {'pytorch_name': 'conv3.weight', 'tf_shape': (3, 3, 16, 16)}, # TF shape (H, W, In, Out)
    'sequential/dense/bias:0': {'pytorch_name': 'fc.bias', 'tf_shape': (10,)},
    'sequential/dense/kernel:0': {'pytorch_name': 'fc.weight', 'tf_shape': (16, 10)} # TF shape (In, Out)
}


class SimpleCNN(nn.Module):
    """Simple CNN architecture compatible with the provided TF parameter mapping."""
    def __init__(self):
        super().__init__() # More compact super() call
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2) # Concise init
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10) # Concise init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_state_dictionary(weight_vector: np.ndarray) -> Dict[str, torch.Tensor]:
    """
    Reconstructs a PyTorch state dictionary from a flattened weight vector
    using a layout DataFrame and a parameter mapping dictionary.

    This function handles the reshaping and permutation required to convert
    TensorFlow-shaped parameters (from the vector) into PyTorch-shaped tensors.

    Args:
        weight_vector (np.ndarray): A flattened NumPy array containing all model weights.

    Returns:
        Dict[str, torch.Tensor]: A PyTorch state dictionary suitable for loading into a model.
    """
    # Extract parameter slices from the flattened vector based on layout_df
    parameter_dict: Dict[str, np.ndarray] = {}
    if not layout_df.empty:
        for _, row in layout_df.iterrows():
            varname = row['varname']
            start_idx = row['start_idx']
            end_idx = row['end_idx']
            # Ensure indices are within bounds
            if 0 <= start_idx < end_idx <= len(weight_vector):
                 parameter_dict[varname] = weight_vector[start_idx:end_idx]
            else:
                 print(f"Warning: Invalid indices for {varname}. Skipping.")

    reconstructed_state_dict: Dict[str, torch.Tensor] = {}

    # Reshape and permute parameters according to param_mapping_and_shape
    for tf_name, flat_np_array in parameter_dict.items():
        mapping_info = param_mapping_and_shape.get(tf_name)
        if mapping_info is None:
            # Skip parameters not included in the mapping (e.g., optimizer states if present)
            continue

        pytorch_name = mapping_info['pytorch_name']
        tf_shape = mapping_info['tf_shape']
        param_tensor = torch.from_numpy(flat_np_array).float()

        # --- Reshaping and Permutation Logic (Improved Readability) ---
        reshaped_param: Optional[torch.Tensor] = None
        expected_pytorch_shape: Optional[tuple] = None

        if 'kernel' in tf_name and 'conv2d' in tf_name:
            # Convolutional kernel: TF (H, W, In, Out) -> PyTorch (Out, In, H, W)
            # Reshape to TF shape first, then permute
            if param_tensor.numel() == np.prod(tf_shape): # Ensure flat size matches TF shape
                 reshaped_param = param_tensor.view(tf_shape).permute(3, 2, 0, 1)
                 expected_pytorch_shape = (tf_shape[3], tf_shape[2], tf_shape[0], tf_shape[1])
            else:
                 print(f"Warning: Flat array size mismatch for {tf_name}. Expected {np.prod(tf_shape)}, got {param_tensor.numel()}. Skipping.")


        elif 'kernel' in tf_name and 'dense' in tf_name:
            # Dense kernel (weight): TF (In, Out) -> PyTorch (Out, In)
            # Reshape to TF shape first, then transpose
            if param_tensor.numel() == np.prod(tf_shape): # Ensure flat size matches TF shape
                 reshaped_param = param_tensor.view(tf_shape).T
                 expected_pytorch_shape = (tf_shape[1], tf_shape[0])
            else:
                 print(f"Warning: Flat array size mismatch for {tf_name}. Expected {np.prod(tf_shape)}, got {param_tensor.numel()}. Skipping.")


        elif 'bias' in tf_name:
            # Bias: Shape is typically the same (1D vector)
            # No explicit reshape needed unless tf_shape enforces it
             if param_tensor.numel() == np.prod(tf_shape): # Ensure flat size matches TF shape
                reshaped_param = param_tensor.view(tf_shape) # view is like reshape here for 1D
                expected_pytorch_shape = tf_shape # Shape is the same for biases
             else:
                 print(f"Warning: Flat array size mismatch for {tf_name}. Expected {np.prod(tf_shape)}, got {param_tensor.numel()}. Skipping.")

        else:
             print(f"Warning: Unhandled parameter type/name pattern for {tf_name}. Skipping.")


        if reshaped_param is not None:
             # Optional: Add assertion or warning if shapes don't match the expected PyTorch shape
             if reshaped_param.shape != expected_pytorch_shape:
                 print(f"Shape mismatch after reshape/permute for {pytorch_name}: expected {expected_pytorch_shape}, got {reshaped_param.shape}. This might cause issues.")
             reconstructed_state_dict[pytorch_name] = reshaped_param

    return reconstructed_state_dict


def convert_weight_vector_to_model(weight_vector: np.ndarray) -> SimpleCNN:
    """
    Converts a flattened weight vector into a PyTorch SimpleCNN model.
    """
    model = SimpleCNN()
    reconstructed_state_dict = create_state_dictionary(weight_vector)
    model.load_state_dict(reconstructed_state_dict)
    return model


def get_cifar10_grayscale_data_loader(batch_size: int = 100, data_root: str = './data', train: bool = False) -> Optional[torch.utils.data.DataLoader]:
    """
    Loads the CIFAR-10 dataset (train or test split), applies grayscale and [-1, 1] normalization,
    and returns a PyTorch DataLoader.

    Args:
        batch_size (int): Batch size for the DataLoader.
        data_root (str): Directory to download/load the dataset from.
        train (bool): If True, loads the training dataset. If False, loads the test dataset.

    Returns:
        torch.utils.data.DataLoader | None: DataLoader for the requested dataset split, or None if loading fails.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Convert to grayscale
        transforms.ToTensor(),                      # Scales to [0, 1]
        transforms.Normalize((0.5,), (0.5,))        # Normalize to [-1, 1]
    ])
    try:
        dataset = torchvision.datasets.CIFAR10(root=data_root, train=train,
                                           download=True, transform=transform)
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        print(f"Please ensure you have internet access to download or the data exists in {data_root}")
        return None
    # Using pin_memory=True can speed up data transfer to GPU if available
    # Shuffle training data, do not shuffle test data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=train, num_workers=2, pin_memory=True)
    return data_loader


def evaluate_model(model: SimpleCNN, dataset_loader: Optional[torch.utils.data.DataLoader]) -> float:
    """
    Evaluates the model on the given test loader.

    Args:
        model (SimpleCNN): The PyTorch model.
        dataset_loader (torch.utils.data.DataLoader | None): The DataLoader.

    Returns:
        float: The accuracy of the model on the test set (as a percentage).
    """
    if dataset_loader is None:
        print("DataLoader is None. Cannot evaluate.")
        return 0.0

    model.eval() # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for images, labels in dataset_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)
            model.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def generate_last_layer_activations(weight_vector: np.ndarray, data_loader: Optional[torch.utils.data.DataLoader]):
    """
    Generates a dataset of the last layer activations (output of global_avg_pool)
    and corresponding labels for the given weight vector and data.

    Args:
        weight_vector (np.ndarray): A flattened NumPy array containing all model weights.
        data_loader (torch.utils.data.DataLoader | None): The DataLoader for the dataset (train or test).
    """
    if data_loader is None:
        print("Data DataLoader is None. Cannot generate activations.")
        return None, None

    model = convert_weight_vector_to_model(weight_vector)
    model.eval() # Set the model to evaluation mode
    all_activations = []

    with torch.no_grad(): # Disable gradient calculation
        for images, labels in data_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            model.to(device) # Move model to device once before the loop

            activations = model.forward(images)
            all_activations.append(activations.cpu().numpy()) # Move to CPU and convert to NumPy

    # Concatenate all batches
    final_activations = np.concatenate(all_activations, axis=0)

    return final_activations