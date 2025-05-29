import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import CNN_evaluator

def save_activations_as_tensors(activations, output_dir, filename_prefix="activations"):
    """
    Saves a dictionary of activations (where keys are layer names) as individual
    PyTorch tensor files (.pt) in the specified output directory.

    Args:
        activations (dict): A dictionary where keys are layer names (strings)
                            and values are PyTorch tensors representing the activations.
        output_dir (str): The directory where the activation tensors will be saved.
        filename_prefix (str): The prefix to use for the saved filenames.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    for layer_name, tensor in activations.items():
        filename = os.path.join(output_dir, f"{filename_prefix}_{layer_name}.pt")
        try:
            torch.save(tensor, filename)
            print(f"Activations for layer '{layer_name}' saved to '{filename}'")
        except Exception as e:
            print(f"Error saving activations for layer '{layer_name}': {e}")

def main():
    """
    Main function to load weights, generate activations, and save them.
    """
    # Define output directory
    output_directory = "activations"

    # Load the optimal weights
    try:
        weights = np.load("optimal_weights.npy")
    except FileNotFoundError:
        print("Error: optimal_weights.npy not found. Please make sure the file exists.")
        return

    # Get the data loaders
    train_loader = CNN_evaluator.get_cifar10_grayscale_data_loader(train=True)
    test_loader = CNN_evaluator.get_cifar10_grayscale_data_loader(train=False)

    # Generate activations
    train_activations = CNN_evaluator.generate_last_layer_activations(weights, train_loader)
    test_activations = CNN_evaluator.generate_last_layer_activations(weights, test_loader)

    # Save the activations as PyTorch tensors
    save_activations_as_tensors(train_activations, output_directory, filename_prefix="train_activations")
    save_activations_as_tensors(test_activations, output_directory, filename_prefix="test_activations")

    print(f"\nActivations saved to the '{output_directory}' directory.")

if __name__ == "__main__":
    main()