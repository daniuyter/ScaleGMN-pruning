import torch
import torch_geometric
from torch_geometric.data import Data
import yaml
import torch.nn.functional as F
import torch.nn as nn
import os
from src.data import dataset
from tqdm import tqdm
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.data.data_utils import get_node_types, get_edge_types
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb
import copy

import torchvision
from torch import optim
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.cifar10_dataset import cnn_to_tg_data
import pandas as pd
import collections
import time

from prune_utils import load_model_layout, unflatten_weights,  generate_static_graph_attributes, calculate_pruning_loss, evaluate_pruned_model, PrunedCNN, run_correlation_check, get_scalegmn_prediction, fine_tune_pruned_model, magnitude_prune_weights, transform_weights_biases, run_gradual_pruning_evaluation


def get_cifar10_loaders(batch_size=128, grayscale=True):
    """ Returns CIFAR-10 train and validation data loaders with optimizations. """
    # print("INFO: Preparing CIFAR-10 data loaders.") # Less verbose
    transform_list = [transforms.ToTensor()]
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    try:
        num_w = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
        if num_w == 0: num_w = 1
    except AttributeError:
        num_w = 4 # Fallback


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                              num_workers=num_w, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                             num_workers=num_w, pin_memory=True)
    # print("INFO: CIFAR-10 data loaders ready.") # Less verbose
    return trainloader, testloader

def apply_global_magnitude_pruning(original_weights, original_biases, target_sparsity):
    """
    Applies global magnitude pruning to the weights. Biases are typically not pruned.
    Returns new lists of pruned weight and bias tensors.
    """
    print(f"INFO: Applying global magnitude pruning for target sparsity: {target_sparsity:.2f}")
    if target_sparsity == 0.0:
        return [w.clone() for w in original_weights], [b.clone() for b in original_biases]

    all_weights_flat = torch.cat([w.flatten() for w in original_weights])
    if all_weights_flat.numel() == 0:
        return [w.clone() for w in original_weights], [b.clone() for b in original_biases]

    k = int(target_sparsity * all_weights_flat.numel())
    if k >= all_weights_flat.numel(): # Avoid trying to prune all weights
        k = all_weights_flat.numel() -1
    
    threshold = torch.kthvalue(torch.abs(all_weights_flat), k + 1 if k < all_weights_flat.numel() else all_weights_flat.numel() ).values.item()

    pruned_weights = []
    for w_tensor in original_weights:
        mask = torch.abs(w_tensor) > threshold
        pruned_w = w_tensor.clone()
        pruned_w[~mask] = 0.0
        pruned_weights.append(pruned_w)

    pruned_biases = [b.clone() for b in original_biases]
    return pruned_weights, pruned_biases

def train_one_epoch(model, optimizer, criterion, train_loader, device, masks, current_fine_tune_epoch, total_fine_tune_epochs):
    """ Trains the model for one epoch. """
    model.train()
    for i, data in enumerate(train_loader): 
        inputs, labels = data 
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if name in masks and param.grad is not None:
                param.grad.data.mul_(masks[name])
        optimizer.step()

def calculate_parameter_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    return zero_params / total_params if total_params > 0 else 0.0

def load_smallcnnzoo_network_weights(base_zoo_weights_path, network_idx, model_layout):
    all_flat_weights = np.load(base_zoo_weights_path, allow_pickle=True)
    if not (0 <= network_idx < len(all_flat_weights)):
        raise ValueError(f"Network index {network_idx} out of bounds for weights file with {len(all_flat_weights)} networks.")
    flat_weights_for_net = all_flat_weights[network_idx]
    # `model_layout` is passed to `unflatten_weights` for its use.
    return unflatten_weights(flat_weights_for_net, model_layout)

# User's provided load_and_populate_cnn, with corrections
def load_and_populate_cnn(PrunedCNN_class, pruned_weights, pruned_biases, 
                          tf_to_pytorch_mapping, device, global_model_layout):
    """
    Instantiates a PrunedCNN model and populates it with the given pruned weights and biases.
    Uses tf_to_pytorch_mapping to match TF layer names to PyTorch layer parameters.
    `global_model_layout` is the layout loaded at the start of generate_pruning_dataset.
    """
    print("INFO: Instantiating and populating CNN model (user's provided logic).")
    model = PrunedCNN_class().to(device)
    
    new_state_dict = collections.OrderedDict()
    weight_idx = 0
    bias_idx = 0
    
    # Ensure global_model_layout is a list of dicts with 'varname'
    if not global_model_layout or not isinstance(global_model_layout, list) or \
       not all(isinstance(item, dict) and 'varname' in item for item in global_model_layout):
        print("ERROR: `global_model_layout` is not in the expected format (list of dicts with 'varname'). Cannot populate model.")
        return None

    # Use the global_model_layout passed as an argument
    for layer_info in global_model_layout: 
        tf_name = layer_info['varname'] # Assumes 'varname' key exists
        pytorch_name = tf_to_pytorch_mapping.get(tf_name) 
    
        if pytorch_name:
            if pytorch_name not in model.state_dict():
                print(f"Warning: PyTorch name {pytorch_name} (from TF: {tf_name}) not in model's state_dict. Skipping.")
                continue
            target_shape = model.state_dict()[pytorch_name].shape 
            
            if 'kernel' in tf_name or '.weight' in pytorch_name: 
                if weight_idx < len(pruned_weights):
                    param_tensor = pruned_weights[weight_idx]
                    if param_tensor.shape == target_shape:
                        # print(f"Mapping {tf_name} -> {pytorch_name}, shape: {param_tensor.shape}")
                        new_state_dict[pytorch_name] = param_tensor
                    else:
                        print(f"!! SHAPE MISMATCH for {pytorch_name} (TF: {tf_name}): Pruned is {param_tensor.shape}, Model expects {target_shape}")
                    weight_idx += 1
                else: print(f"Warning: Ran out of pruned weights for {tf_name}")
            elif 'bias' in tf_name or '.bias' in pytorch_name: 
                if bias_idx < len(pruned_biases):
                    param_tensor = pruned_biases[bias_idx]
                    if param_tensor is not None:
                         if param_tensor.shape == target_shape:
                             # print(f"Mapping {tf_name} -> {pytorch_name}, shape: {param_tensor.shape}")
                             new_state_dict[pytorch_name] = param_tensor
                         else:
                              print(f"!! SHAPE MISMATCH for {pytorch_name} (TF: {tf_name}): Pruned is {param_tensor.shape}, Model expects {target_shape}")
                    else: print(f"Skipping mapping for {tf_name} (bias was None)") # Should not happen if unflatten_weights is correct
                    bias_idx += 1
                else: print(f"Warning: Ran out of pruned biases for {tf_name}")
        else: print(f"Warning: No PyTorch mapping found for TF name {tf_name}")
        
    try:
        # It's safer to load into the existing state_dict or ensure all keys are present
        # For simplicity here, we try to load the new_state_dict.
        # This might fail if not all parameters of the model are in new_state_dict.
        # A more robust way is to update model.state_dict().
        current_model_state_dict = model.state_dict()
        current_model_state_dict.update(new_state_dict)
        model.load_state_dict(current_model_state_dict)
        print("\nSuccessfully loaded pruned weights into CNN model using user's logic.")
    except RuntimeError as e:
        print(f"\nERROR loading state_dict with user's logic: {e}")
        print("Check CNN architecture, parameter mapping, and if all model parameters were assigned in new_state_dict.")
        return None # Return None on failure
    return model

# User's provided fine_tune_cnn
def fine_tune_cnn(cnn_instance, train_loader, val_loader, epochs, learning_rate, device):
    """
    Fine-tunes the given CNN instance.
    Crucially, it should respect the pruning mask (zeroed weights should stay zero).
    """
    print(f"INFO: Starting fine-tuning for {epochs} epochs with LR: {learning_rate}.")
    cnn_instance.train()
    optimizer = optim.Adam(cnn_instance.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    masks = {}
    for name, param in cnn_instance.named_parameters():
        if 'weight' in name: 
            masks[name] = (param.data != 0).float()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} FT", leave=False)):
            inputs, labels = data # data is already a tuple (inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnn_instance(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in cnn_instance.named_parameters():
                if name in masks and param.grad is not None:
                    param.grad.data.mul_(masks[name])
            
            optimizer.step()
            running_loss += loss.item()
        # print(f"INFO: Epoch {epoch+1} FT Loss: {running_loss / len(train_loader):.4f}") # Optional: per-epoch loss

    accuracy = evaluate_pruned_model(cnn_instance, val_loader, device, prefix="Post-FT")
    # print(f"INFO: Fine-tuning complete. Post-FT Accuracy: {accuracy:.2f}%") # Printed by evaluate_pruned_model
    return accuracy

# User's provided evaluate_pruned_model
def evaluate_pruned_model(model, val_loader, device, prefix="Evaluation"):
    """Evaluates the pruned model on the validation set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"{prefix} Evaluation", leave=False):
            images, labels = data # data is already a tuple
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"INFO: {prefix} Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main Dataset Generation Function (Hardcoded Parameters) ---
def generate_pruning_dataset(): 
    HARDCODED_BASE_ZOO_WEIGHTS_PATH = './data/cifar10/weights.npy' 
    HARDCODED_METADATA_CSV_PATH = './data/cifar10/metrics.csv.gz'     
    HARDCODED_NETWORK_ID_COLUMN_IN_CSV = 'network_id'
    HARDCODED_ACTIVATION_COLUMN_IN_CSV = 'config.activation' 
    HARDCODED_TARGET_ACTIVATION = 'relu'
    HARDCODED_LAYOUT_PATH = '/home/scur2670/scalegmn/data/cifar10/layout.csv' 
    HARDCODED_OUTPUT_DIR = './dataset_pruned' 
    HARDCODED_NUM_NETWORKS_TO_PROCESS = 10 
    HARDCODED_SPARSITY_LEVELS = np.linspace(0.0, 0.5, num=50).tolist()
    TOTAL_FINETUNE_EPOCHS = 10
    HARDCODED_FINETUNE_LR = 1e-4
    TARGET_TOTAL_PRUNED_INSTANCES = 1350
    TARGET_TOTAL_BASE_INSTANCE_COMBINATIONS = 1350 
    HARDCODED_PRETRAINED_ACC_COLUMN_IN_CSV = 'test_accuracy'
    EPOCHS_TO_SAVE_AT = [0, 5, 10]

    HARDCODED_TF_TO_PYTORCH_MAPPING = {
        'sequential/conv2d/kernel:0':   'features.0.weight',
        'sequential/conv2d/bias:0':     'features.0.bias',
        'sequential/conv2d_1/kernel:0': 'features.3.weight',
        'sequential/conv2d_1/bias:0':   'features.3.bias',
        'sequential/conv2d_2/kernel:0': 'features.6.weight',
        'sequential/conv2d_2/bias:0':   'features.6.bias',
        'sequential/dense/kernel:0':    'fc.weight',
        'sequential/dense/bias:0':      'fc.bias',
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    global_model_layout = load_model_layout(HARDCODED_LAYOUT_PATH) 
    train_loader, val_loader = get_cifar10_loaders(grayscale=True)
    
    if not os.path.exists(HARDCODED_BASE_ZOO_WEIGHTS_PATH):
        print(f"ERROR: Weights file not found at {HARDCODED_BASE_ZOO_WEIGHTS_PATH}")
        return
    all_base_weights = np.load(HARDCODED_BASE_ZOO_WEIGHTS_PATH, allow_pickle=True)
    total_networks_in_zoo = len(all_base_weights)
    print(total_networks_in_zoo)

    
    df_metadata_full = None

    valid_network_indices = []
    if not os.path.exists(HARDCODED_METADATA_CSV_PATH):
        print(f"WARNING: Metadata CSV not found at {HARDCODED_METADATA_CSV_PATH}. Processing all {total_networks_in_zoo} networks.")
        valid_network_indices = list(range(total_networks_in_zoo))

    else:
        try:
            compression = 'gzip' if HARDCODED_METADATA_CSV_PATH.endswith('.gz') else None
            df_metadata_full = pd.read_csv(HARDCODED_METADATA_CSV_PATH, compression=compression) 

            # we dont have column network ids, its the first column of the df_metadata
            act_series_from_csv = df_metadata_full['config.activation']
            
            # Filter based on activation
            # Important: This creates a boolean Series aligned with df_metadata's index
            activation_mask = (act_series_from_csv == 'relu')
            
            # Get the IDs (which are indices for all_base_weights) that satisfy the activation criteria
            all_indices = range(len(df_metadata_full))
            csv_indices_meeting_activation = list([all_indices[i] for i in range(len(all_indices)) if activation_mask[i]])

            # Further filter these to ensure they are valid indices for the loaded all_base_weights
            valid_network_indices = [idx for idx in csv_indices_meeting_activation if 0 <= idx < total_networks_in_zoo]
            
            
            print(f"INFO: Found {activation_mask.sum()} networks with activation '{HARDCODED_TARGET_ACTIVATION}' in CSV. "
                  f"{len(valid_network_indices)} of these are valid indices for the loaded weights.")
    
            if not valid_network_indices:
                print("ERROR: No networks found matching the target activation filter and also being valid indices. Exiting.")
                return

        except Exception as e:
            print(f"ERROR: Failed to process metadata CSV: {e}")
            import traceback
            traceback.print_exc()
            return

    os.makedirs(HARDCODED_OUTPUT_DIR, exist_ok=True)
    dataset_log = []

    print(f"INFO: Starting generation of {TARGET_TOTAL_BASE_INSTANCE_COMBINATIONS} (base_net, sparsity) combinations.")
    print(f"      Each combination will be fine-tuned for {TOTAL_FINETUNE_EPOCHS} epochs.")
    print(f"      Models will be saved at epochs: {EPOCHS_TO_SAVE_AT}")
    
    main_pbar = tqdm(range(TARGET_TOTAL_BASE_INSTANCE_COMBINATIONS), desc="Processing Base Instances")
    for i in main_pbar:
        start_time = time.time()
        base_net_idx = random.choice(valid_network_indices)
        sparsity_level = random.choice(HARDCODED_SPARSITY_LEVELS)
        
        main_pbar.set_postfix_str(f"Current: NetID={base_net_idx}, Sparsity={sparsity_level:.3f}")
        print(f"\n---> Processing Instance {i+1}/{TARGET_TOTAL_BASE_INSTANCE_COMBINATIONS}: NetID={base_net_idx}, Sparsity={sparsity_level:.4f} <---")

        pretrained_accuracy = -1.0 
        if df_metadata_full is not None and base_net_idx in df_metadata_full.index:
            try: pretrained_accuracy = df_metadata_full.loc[base_net_idx, HARDCODED_PRETRAINED_ACC_COLUMN_IN_CSV]
            except KeyError: print(f"Warning: Pre-trained acc not found for index {base_net_idx}.")
        print(f"INFO: Base Net ID: {base_net_idx}, Pre-trained Acc (from CSV): {pretrained_accuracy:.2f}%")

        try:
            print(f"INFO: Loading original weights for Net ID {base_net_idx}...")
            original_weights, original_biases = unflatten_weights(all_base_weights[base_net_idx], global_model_layout, device)
            if not original_weights and not any(b is not None for b in original_biases if original_biases): 
                 print(f"WARNING: Skipping instance (Net ID {base_net_idx}) due to empty weights/biases from loader.")
                 continue
            print(f"INFO: Original weights loaded. Applying pruning for sparsity {sparsity_level:.4f}...")
            pruned_weights, pruned_biases = apply_global_magnitude_pruning(
                original_weights, original_biases, sparsity_level
            )
            print(f"INFO: Pruning applied. Populating CNN instance...")
            cnn_instance = load_and_populate_cnn(
                PrunedCNN, pruned_weights, pruned_biases, 
                HARDCODED_TF_TO_PYTORCH_MAPPING, device, global_model_layout
            )
            if cnn_instance is None: 
                print(f"ERROR: Failed to load/populate CNN for (Net ID {base_net_idx}, Sparsity {sparsity_level:.2f}). Skipping.")
                continue
            print(f"INFO: CNN instance populated.")
            
            actual_initial_sparsity = calculate_parameter_sparsity(cnn_instance)
            print(f"INFO: Actual initial sparsity before fine-tuning: {actual_initial_sparsity:.4f}")

            if 0 in EPOCHS_TO_SAVE_AT:
                print(f"  INFO: Reached save point at Epoch 0 (pre-fine-tuning). Evaluating and saving model...")
                eval_prefix_epoch0 = f"NetID={base_net_idx}, Sparsity={sparsity_level:.2f}, Epoch=0"
                accuracy_epoch0 = evaluate_pruned_model(cnn_instance, val_loader, device, prefix=eval_prefix_epoch0)
                sparsity_epoch0 = actual_initial_sparsity # Sparsity is the same as initial
                print(f"  INFO (Epoch 0 End): Accuracy={accuracy_epoch0:.2f}%, Sparsity={sparsity_epoch0:.4f}")

                output_filename_epoch0 = f"net_{base_net_idx}_sparsity_{sparsity_level*10000:.0f}_epoch_0.pt" 
                output_path_epoch0 = os.path.join(HARDCODED_OUTPUT_DIR, output_filename_epoch0)
                
                saved_data_epoch0 = {
                    'original_network_id': base_net_idx,
                    'target_sparsity': sparsity_level,
                    'epoch': 0,
                    'pretrained_accuracy': pretrained_accuracy,
                    'actual_initial_sparsity_before_ft': actual_initial_sparsity, 
                    'sparsity_at_epoch_end': sparsity_epoch0,
                    'accuracy_at_epoch_end': accuracy_epoch0,
                    'model_state_dict': cnn_instance.state_dict(),
                    'activation_used': HARDCODED_TARGET_ACTIVATION,
                    'model_layout_path': HARDCODED_LAYOUT_PATH, 
                    'pruned_cnn_architecture': PrunedCNN.__name__
                }
                torch.save(saved_data_epoch0, output_path_epoch0)
                print(f"  INFO: Saved model to {output_path_epoch0}")
                
                dataset_log.append({
                    'original_network_id': base_net_idx, 
                    'target_sparsity': sparsity_level,
                    'epoch': 0,
                    'pretrained_accuracy': pretrained_accuracy,
                    'actual_initial_sparsity_before_ft': actual_initial_sparsity,
                    'sparsity_at_epoch_end': sparsity_epoch0,
                    'accuracy_at_epoch_end': accuracy_epoch0,
                    'saved_file': output_path_epoch0,
                    'activation_used': HARDCODED_TARGET_ACTIVATION
                })

            optimizer = optim.Adam(cnn_instance.parameters(), lr=HARDCODED_FINETUNE_LR)
            criterion = nn.CrossEntropyLoss()
            masks = {name: (param.data != 0).float() for name, param in cnn_instance.named_parameters() if 'weight' in name}

            print(f"INFO: Starting fine-tuning for {TOTAL_FINETUNE_EPOCHS} epochs...")
            for epoch_num in range(TOTAL_FINETUNE_EPOCHS):
                current_epoch_display = epoch_num + 1
                print(f"  Fine-tuning Epoch {current_epoch_display}/{TOTAL_FINETUNE_EPOCHS} for NetID={base_net_idx}, Sparsity={sparsity_level:.4f}")
                train_one_epoch(cnn_instance, optimizer, criterion, train_loader, device, masks, current_epoch_display, TOTAL_FINETUNE_EPOCHS)
                
                # Save model only at specified epochs
                if current_epoch_display in EPOCHS_TO_SAVE_AT:
                    print(f"  INFO: Reached save point at Epoch {current_epoch_display}. Evaluating and saving model...")
                    eval_prefix = f"NetID={base_net_idx}, Sparsity={sparsity_level:.2f}, Epoch={current_epoch_display}"
                    current_epoch_accuracy = evaluate_pruned_model(cnn_instance, val_loader, device, prefix=eval_prefix)
                    current_epoch_sparsity = calculate_parameter_sparsity(cnn_instance) 
                    print(f"  INFO (Epoch {current_epoch_display} End): Accuracy={current_epoch_accuracy:.2f}%, Sparsity={current_epoch_sparsity:.4f}")

                    output_filename = f"net_{base_net_idx}_sparsity_{sparsity_level*10000:.0f}_epoch_{current_epoch_display}.pt" 
                    output_path = os.path.join(HARDCODED_OUTPUT_DIR, output_filename)
                    
                    saved_data = {
                        'original_network_id': base_net_idx,
                        'target_sparsity': sparsity_level,
                        'epoch': current_epoch_display,
                        'pretrained_accuracy': pretrained_accuracy,
                        'actual_initial_sparsity_before_ft': actual_initial_sparsity, 
                        'sparsity_at_epoch_end': current_epoch_sparsity,
                        'accuracy_at_epoch_end': current_epoch_accuracy,
                        'model_state_dict': cnn_instance.state_dict(),
                        'activation_used': HARDCODED_TARGET_ACTIVATION,
                        'model_layout_path': HARDCODED_LAYOUT_PATH, 
                        'pruned_cnn_architecture': PrunedCNN.__name__
                    }
                    torch.save(saved_data, output_path)
                    print(f"  INFO: Saved model to {output_path}")
                    
                    dataset_log.append({
                        'original_network_id': base_net_idx, 
                        'target_sparsity': sparsity_level,
                        'epoch': current_epoch_display,
                        'pretrained_accuracy': pretrained_accuracy,
                        'actual_initial_sparsity_before_ft': actual_initial_sparsity,
                        'sparsity_at_epoch_end': current_epoch_sparsity,
                        'accuracy_at_epoch_end': current_epoch_accuracy,
                        'saved_file': output_path,
                        'activation_used': HARDCODED_TARGET_ACTIVATION
                    })
            print(f"INFO: Fine-tuning complete for NetID={base_net_idx}, Sparsity={sparsity_level:.4f}")
            end_time = time.time()
            print(start_time - end_time, 'seconds elapsed')
        except Exception as e:
            print(f"ERROR: Unhandled exception for (Net ID {base_net_idx}, Sparsity {sparsity_level:.2f}): {e}")
            traceback.print_exc()
            continue



    log_df = pd.DataFrame(dataset_log)
    log_path = os.path.join(HARDCODED_OUTPUT_DIR, "_dataset_generation_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"INFO: Dataset generation complete. Log saved to {log_path}")

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args() 
    generate_pruning_dataset()