import torch
import pandas as pd
import numpy as np
import collections
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn 

from src.data.data_utils import get_node_types, get_edge_types
from src.data.cifar10_dataset import pad_and_flatten_kernel
from src.data.cifar10_dataset import cnn_to_tg_data
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.cifar10_dataset import cnn_to_tg_data
import pandas as pd
import collections
import time


def run_gradual_pruning_evaluation(
    original_weights_cpu, original_biases_cpu, # Starting point
    scalegmn_net, # Pre-loaded ScaleGMN
    model_layout, tf_to_pytorch_mapping, # For loading CNN
    cnn_config, # Activation, dropout etc. for eval model
    static_graph_attrs, # For ScaleGMN input
    device, conf): # Main config
    """
    Runs a gradual pruning process, stopping at sparsity intervals
    to evaluate ScaleGMN prediction vs actual fine-tuned accuracy.
    """
    print("\n--- Starting Gradual Pruning Evaluation ---")

    gradual_conf = conf.get("gradual_pruning_eval", {})
    max_epochs = gradual_conf.get("max_epochs", 30000)
    sparsity_interval = gradual_conf.get("sparsity_interval", 10.0) # Check every 20%
    finetune_epochs_ckpt = gradual_conf.get("finetune_epochs_ckpt", 50) # FT epochs at checkpoint
    finetune_lr_ckpt = gradual_conf.get("finetune_lr_ckpt", 1e-3) # FT LR at checkpoint

    # --- Setup for the gradual pruning run ---
    pruning_conf = conf['pruning_args'] # Use settings from main pruning section
    alpha = pruning_conf.get('alpha', 1e-4)
    beta = pruning_conf.get('beta', 1.0) # Weight for ScaleGMN predictor in loss
    pruning_threshold = pruning_conf.get('threshold', 1e-4) # Hard threshold after step

    # Initialize weights/biases on device, requiring grad
    raw_weights = [p.clone().to(device).requires_grad_(True) for p in original_weights_cpu]
    raw_biases = [(p.clone().to(device).requires_grad_(True) if p is not None else None) for p in original_biases_cpu]
    params_to_optimize = raw_weights + [p for p in raw_biases if p is not None]
    if not params_to_optimize:
        print("ERROR: No parameters found to optimize.")
        return

    conf_opt = conf['optimization']
    optimizer, scheduler = setup_optimization(params_to_optimize, optimizer_name=conf_opt['optimizer_name'],
                                              optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])

    print(f"Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler[0]).__name__ if scheduler else 'None'}")

    # Data loaders needed for fine-tuning checkpoints
    batch_size = 256
    num_workers = 8
    train_loader, val_loader, test_loader = get_cifar10_loaders(
         batch_size=batch_size,
         num_workers=num_workers
    )
    if train_loader is None or val_loader is None or test_loader is None:
        print("ERROR: Failed to get data loaders for gradual evaluation.")
        return

    # --- Tracking Results ---
    results = {
        'epoch': [],
        'sparsity': [],
        'predicted_acc': [],
        'actual_acc_ft': [],
        'total_loss': [],
        'l1_loss': [],
        'pred_acc_loss_term': [] # Store the pred_acc used in loss calculation
    }
    next_sparsity_target = 60 # Start checking at the first interval
    last_sparsity_check = -1.0 # To avoid redundant checks if sparsity stalls

    print(f"Running up to {max_epochs} epochs, evaluating every {sparsity_interval}% sparsity increase.")
    start_time = time.time()

    # --- Gradual Pruning Loop ---
    for epoch in tqdm(range(120000)):
        
        epoch_start_time = time.time()
        optimizer.zero_grad()

        # Calculate pruning loss (using current weights on device)
        total_loss, l1_loss, pred_acc_loss_term = calculate_pruning_loss(
            raw_weights, raw_biases, scalegmn_net, alpha, beta, device,
            transform_weights_biases, cnn_to_tg_data, # Pass function handles
                static_graph_attrs["conv_mask"],
                static_graph_attrs["direction"],
                static_graph_attrs["layer_layout"],
                static_graph_attrs["mask_hidden"],
                static_graph_attrs["node2type"],
                static_graph_attrs["edge2type"],
                static_graph_attrs["fmap_size"],
                static_graph_attrs["sign_mask"],
                # --- Other args ---
                static_graph_attrs["max_kernel_size"]
            # max_kernel_size=static_graph_attrs["max_kernel_size"] # Pass if needed
        )

        # Optimize
       
        total_loss.backward()

        # --- Apply mask to gradients of zeroed-out parameters ---
        with torch.no_grad():
            for param in params_to_optimize: # params_to_optimize are raw_weights & raw_biases
                if param.grad is not None:
                    # Create a mask for elements that are ALREADY zero in the parameter tensor
                    already_zero_mask = (param.data == 0.0)
                    # Set the gradient to zero for these elements
                    param.grad.data[already_zero_mask] = 0.0
        # --- End gradient masking ---
        
        optimizer.step()

        threshold = 1e-4
        # Perform thresholding without tracking gradients
        with torch.no_grad():
            # Iterate through the same list of tensors the optimizer is updating
            for param in params_to_optimize:
                if param is not None:
                    # Create a mask for elements below the threshold
                    # Use .data for in-place modification to avoid issues with leaf variables
                    mask = param.data.abs() < threshold
                    # Set those elements to exactly 0.0
                    param.data[mask] = 0.0

        if scheduler is not None and scheduler[0] is not None: # Check scheduler object itself
            if scheduler[1] == 'ReduceLROnPlateau':
                 if torch.isfinite(total_loss):
                     scheduler[0].step(total_loss.item()) # Pass scalar loss
                 else:
                     print(f"Skipping scheduler step for epoch {epoch} due to non-finite loss.")
            else:
                 scheduler[0].step()

        
        current_sparsity = calculate_parameter_sparsity(raw_weights + [b for b in raw_biases if b is not None])

        print(f"Epoch {epoch+1}/{max_epochs} | Sparsity: {current_sparsity:.2f}% | Loss: {total_loss.item():.4f} | L1 Loss: {l1_loss:.4f} | Pred Acc Loss Term: {pred_acc_loss_term:.4f}")

        # --- Checkpoint Evaluation ---
        checkpoint_triggered = False
        if current_sparsity >= next_sparsity_target and current_sparsity != last_sparsity_check:
             checkpoint_triggered = True
             print(f"\n--- Checkpoint at Epoch {epoch+1}: Sparsity reached {current_sparsity:.2f}% (Target: {next_sparsity_target:.1f}%) ---")
             checkpoint_start_time = time.time()
            
             # 1. Get copies of current weights (CPU)
             weights_cpu_ckpt = [p.detach().cpu().clone() for p in raw_weights]
             biases_cpu_ckpt = [p.detach().cpu().clone() if p is not None else None for p in raw_biases]

             # 2. Get ScaleGMN Prediction for this state
             pred_acc_ckpt = get_scalegmn_prediction(
                 weights_cpu_ckpt, biases_cpu_ckpt, scalegmn_net,
                 transform_weights_biases, cnn_to_tg_data,
                 static_graph_attrs, device
             )
             print(f"  ScaleGMN Predicted Acc: {pred_acc_ckpt:.4f}")

             # 3. Load into fresh CNN instance
             cnn_ckpt = PrunedCNN(
                 activation_fn=getattr(nn, cnn_config.get("activation", "ReLU"), nn.ReLU),
                 num_classes=cnn_config.get("num_classes", 10),
                 dropout_rate=cnn_config.get("dropout_rate", 0.053518)
             ).to(device)

             state_dict_ckpt = create_pytorch_state_dict(
                 weights_cpu_ckpt, biases_cpu_ckpt, model_layout,
                 tf_to_pytorch_mapping, cnn_ckpt
             )

             load_success = False
             if state_dict_ckpt:
                 try:
                     for key in state_dict_ckpt: state_dict_ckpt[key] = state_dict_ckpt[key].to(device)
                     cnn_ckpt.load_state_dict(state_dict_ckpt, strict=True)
                     load_success = True
                 except Exception as e:
                     print(f"  ERROR loading state dict for checkpoint: {e}")
             else:
                  print(f"  ERROR creating state dict for checkpoint.")

             actual_acc_ckpt = 0.0
             if load_success:
                 # 4. Fine-tune this checkpoint model
                 cnn_ckpt = fine_tune_pruned_model(
                     cnn_ckpt, train_loader, device, finetune_epochs_ckpt, 1e-3, val_loader=val_loader
                 )
                 # 5. Evaluate the fine-tuned checkpoint model
                 actual_acc_ckpt = evaluate_pruned_model(cnn_ckpt, val_loader, device)
                 print(f"  Actual Val Acc (after {finetune_epochs_ckpt} FT epochs): {actual_acc_ckpt*100:.2f}%")
                
                 # 6. Save the model with its sparsity and accuracy in the filename
                 sparsity_str = f"{int(current_sparsity)}"
                 acc_str = f"{actual_acc_ckpt*100:.2f}".replace('.', '_')
                 model_filename = f"model_sparsity_{sparsity_str}_acc_{acc_str}_threshold_{threshold}.pt"
                 torch.save(cnn_ckpt.state_dict(), model_filename)
                 print(f"Saved model to '{model_filename}'")
             else:
                 print("  Skipping fine-tuning and evaluation due to load error.")
             

             # Store results
             results['epoch'].append(epoch + 1)
             results['sparsity'].append(current_sparsity)
             results['predicted_acc'].append(pred_acc_ckpt)
             results['actual_acc_ft'].append(actual_acc_ckpt)
             # Also store loss components for this epoch if desired (might be slightly off if loss calculated before ckpt)
             results['total_loss'].append(total_loss.item())
             results['l1_loss'].append(l1_loss)
             results['pred_acc_loss_term'].append(pred_acc_loss_term)


             if (epoch + 1) % 100 == 0:
                plt.figure(figsize=(12, 8))

                # Sparsity plot
                plt.subplot(3, 1, 1)
                plt.plot(results['epoch'], results['sparsity'], label="Sparsity (%)", color="blue")
                plt.title("Sparsity Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Sparsity (%)")
                plt.grid(True)
                plt.legend()

                # L1 Loss plot
                plt.subplot(3, 1, 2)
                plt.plot(results['epoch'], results['l1_loss'], label="L1 Loss", color="orange")
                plt.title("L1 Loss Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("L1 Loss")
                plt.grid(True)
                plt.legend()

                # Predicted Accuracy plot
                plt.subplot(3, 1, 3)
                plt.plot(results['epoch'], results['predicted_acc'], label="Predicted Accuracy", color="green")
                plt.title("Predicted Accuracy Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Predicted Accuracy")
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.savefig("pruning_progress.png")  # Overwrite the plot file
                plt.close()

                
             # Update target for next checkpoint
             last_sparsity_check = current_sparsity
             next_sparsity_target += sparsity_interval
             print(f"  Checkpoint evaluation took {time.time() - checkpoint_start_time:.2f}s. for sparsity {current_sparsity:.2f}%")

        # Basic progress print
        epoch_time = time.time() - epoch_start_time
        if (epoch + 1) % 50 == 0:
            if not checkpoint_triggered:
                print(f"Epoch {epoch+1}/{max_epochs} | Sparsity: {current_sparsity:.2f}% | Loss: {total_loss.item():.4f} | Time: {epoch_time:.2f}s")


    total_time = time.time() - start_time
    print(f"\n--- Gradual Pruning Evaluation Finished in {total_time:.2f} seconds ---")

    # --- Plotting Results ---
    if not results['epoch']:
        print("No checkpoint data collected to plot.")
        return

    print("\n--- Plotting Gradual Pruning Evaluation Results ---")
    epochs_ckpt = results['epoch']
    sparsities_ckpt = results['sparsity']
    predicted_accs_ckpt = results['predicted_acc']
    actual_accs_ckpt = results['actual_acc_ft']

    # Plot 1: Predicted vs Actual Accuracy
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(predicted_accs_ckpt, actual_accs_ckpt, c=sparsities_ckpt, cmap='viridis', marker='o')
    plt.title('ScaleGMN Predicted vs. Actual Accuracy at Sparsity Milestones')
    plt.xlabel('ScaleGMN Predicted Accuracy (at Checkpoint)')
    plt.ylabel(f'Actual Test Accuracy (after {finetune_epochs_ckpt} FT epochs)')
    plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Achieved Sparsity (%) at Checkpoint')
    # Add correlation if desired


    plt.tight_layout()
    plt.savefig("gradual_eval_pred_vs_actual.png")
    print("Saved plot: gradual_eval_pred_vs_actual.png")
    plt.close() # Close figure

    # Plot 2: Actual Accuracy vs Sparsity
    plt.figure(figsize=(8, 6))
    plt.plot(sparsities_ckpt, actual_accs_ckpt, marker='o', linestyle='-', color='green')
    plt.title(f'Actual Accuracy (after {finetune_epochs_ckpt} FT epochs) vs. Sparsity')
    plt.xlabel('Achieved Sparsity (%) at Checkpoint')
    plt.ylabel(f'Actual Test Accuracy (Fine-tuned)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("gradual_eval_actual_vs_sparsity.png")
    print("Saved plot: gradual_eval_actual_vs_sparsity.png")
    plt.close()

    # Plot 3: Predicted Accuracy vs Sparsity
    plt.figure(figsize=(8, 6))
    plt.plot(sparsities_ckpt, predicted_accs_ckpt, marker='o', linestyle='-', color='blue')
    plt.title('ScaleGMN Predicted Accuracy vs. Sparsity')
    plt.xlabel('Achieved Sparsity (%) at Checkpoint')
    plt.ylabel('ScaleGMN Predicted Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("gradual_eval_pred_vs_sparsity.png")
    print("Saved plot: gradual_eval_pred_vs_sparsity.png")
    plt.close()

    # Plot 4: Sparsity vs Epoch (of checkpoint)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_ckpt, sparsities_ckpt, marker='o', linestyle='-', color='red')
    plt.title('Achieved Sparsity vs. Epoch at Checkpoints')
    plt.xlabel('Epoch Number of Checkpoint')
    plt.ylabel('Achieved Sparsity (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("gradual_eval_sparsity_vs_epoch.png")
    print("Saved plot: gradual_eval_sparsity_vs_epoch.png")
    plt.close()

    
    
    

def create_pytorch_state_dict(parameter_weights, parameter_biases, model_layout, tf_to_pytorch_mapping, target_model):
    """
    
    Args:
        parameter_weights (list): List of weight tensors (op CPU of correct device).
        parameter_biases (list): Lijst of bias tensors of None.
        model_layout (list): De output of load_model_layout.
        tf_to_pytorch_mapping (dict): Mapping van TF varname naar PyTorch layer name.
        target_model (torch.nn.Module): Een instantie van het doel PyTorch model
                                        om shapes te controleren.

    Returns:
        collections.OrderedDict: De gegenereerde state_dict.
                                 Kan leeg zijn als er fouten optreden.
    """
    
    print('parameters weights and biasses shapes')
    print(len(parameter_weights))
    print(len(parameter_biases))
    new_state_dict = collections.OrderedDict()
    weight_idx = 0
    bias_idx = 0
    model_state_dict_keys = target_model.state_dict().keys()

    print(f"Attempting to map {len(parameter_weights)} weights and {len(parameter_biases)} biases...")

    for layer_info in model_layout:
        tf_name = layer_info['varname']
        pytorch_name = tf_to_pytorch_mapping.get(tf_name)

        if pytorch_name:
            if pytorch_name not in model_state_dict_keys:
                print(f"Warning: Mapped PyTorch name '{pytorch_name}' (from TF: '{tf_name}') not found in target model state_dict. Skipping.")
                continue

            try:
                target_shape = target_model.state_dict()[pytorch_name].shape
                is_weight = 'kernel' in tf_name.lower() or 'weight' in pytorch_name.lower()
                is_bias = 'bias' in tf_name.lower()

                if is_weight:
                    if weight_idx < len(parameter_weights):
                        param_tensor = parameter_weights[weight_idx]
                        if param_tensor.shape == target_shape:
                            print(f"  Mapping WEIGHT {tf_name} -> {pytorch_name}, shape: {param_tensor.shape}")
                            new_state_dict[pytorch_name] = param_tensor.clone().detach()
                        else:
                            print(f"!! SHAPE MISMATCH for WEIGHT {pytorch_name}: Pruned/Source is {param_tensor.shape}, Model expects {target_shape}")
                        weight_idx += 1
                    else:
                        print(f"Warning: Ran out of source weights while trying to map {tf_name}")
                elif is_bias:
                    if bias_idx < len(parameter_biases):
                        param_tensor = parameter_biases[bias_idx]
                        if param_tensor is not None:
                            if param_tensor.shape == target_shape:
                                print(f"  Mapping BIAS   {tf_name} -> {pytorch_name}, shape: {param_tensor.shape}")
                                new_state_dict[pytorch_name] = param_tensor.clone().detach() 
                            else:
                                print(f"!! SHAPE MISMATCH for BIAS {pytorch_name}: Pruned/Source is {param_tensor.shape}, Model expects {target_shape}")
                        else:
                             print(f"  Skipping mapping for BIAS {tf_name} (source bias was None)")
                        bias_idx += 1
                    else:
                        print(f"Warning: Ran out of source biases while trying to map {tf_name}")
                else:

                    print(f"Warning: Cannot determine if {tf_name} -> {pytorch_name} is weight or bias.")

            except KeyError:

                print(f"ERROR: PyTorch layer name '{pytorch_name}' suddenly not found in target model state_dict during shape check.")
            except Exception as e:
                 print(f"ERROR during mapping for {tf_name} -> {pytorch_name}: {e}")

        else:

            if 'kernel' in tf_name.lower() or 'bias' in tf_name.lower():
                print(f"Warning: No PyTorch mapping found for potential parameter TF name {tf_name}")

    mapped_keys = set(new_state_dict.keys())
    expected_keys = set(model_state_dict_keys)
    missing_keys = expected_keys - mapped_keys
    if missing_keys:
        print(f"Warning: The following parameters in the target model were NOT mapped: {missing_keys}")
    if weight_idx < len(parameter_weights):
        print(f"Warning: {len(parameter_weights) - weight_idx} source weights were unused.")
    if bias_idx < len(parameter_biases):

         unused_biases = sum(1 for b in parameter_biases[bias_idx:] if b is not None)
         if unused_biases > 0:
             print(f"Warning: {unused_biases} source biases were unused.")


    return new_state_dict
    
    
    
    

def load_model_layout(csv_path):
    try:
        layout_df = pd.read_csv(csv_path)
        model_layout = []
        for _, row in layout_df.iterrows():
            model_layout.append({
                'varname': row['varname'],
                'start_idx': int(row['start_idx']),
                'end_idx': int(row['end_idx']),
                'shape': eval(row['shape']) if isinstance(row['shape'], str) else row['shape']
            })
        print(f"Loaded model layout from {csv_path}")
        return model_layout
    except Exception as e:
        print(f"ERROR: Failed to load or parse layout file {csv_path}: {e}")
        return []


def unflatten_weights(flat_weights_np, model_layout, device):   
    param_dict = {}
    print("\n--- Un-flattening weights ---")
    for layer_info in model_layout:
        varname = layer_info['varname']
        shape = layer_info['shape']
        start_idx = layer_info['start_idx']
        end_idx = layer_info['end_idx']

        param_np = flat_weights_np[start_idx:end_idx]
        expected_elements = np.prod(shape)
        if param_np.size != expected_elements:
             print(f"Warning: Size mismatch for {varname}. Expected {expected_elements}, got {param_np.size}. Reshaping might fail.")
        try:
            param_tensor = torch.from_numpy(param_np).view(shape).to(device)
            param_dict[varname] = param_tensor
            print(f"  Extracted {varname}, shape: {param_tensor.shape}")
        except RuntimeError as e:
            print(f"ERROR reshaping {varname} with shape {shape} from slice size {param_np.size}: {e}")
            param_dict[varname] = torch.zeros(shape, device=device)

    layer_kernels = {}
    layer_biases = {}
    ordered_prefixes = []
    processed_prefixes = set()

    for layer_info in model_layout:
        varname = layer_info['varname']
        parts = varname.rsplit('/', 1)
        if len(parts) == 2:
            prefix, param_type_raw = parts
            param_type = param_type_raw.split(':')[0] 
        else:
            prefix = varname
            param_type = 'unknown'
            if 'bias' in varname.lower(): param_type = 'bias'
            elif 'kernel' in varname.lower() or 'weight' in varname.lower(): param_type = 'kernel'
            print(f"Warning: Could not determine prefix/type reliably for {varname}. Assuming prefix='{prefix}', type='{param_type}'")


        if prefix not in processed_prefixes:
            ordered_prefixes.append(prefix)
            processed_prefixes.add(prefix)

        if varname in param_dict:
            tensor = param_dict[varname]
            if param_type == 'kernel':
                if tensor.ndim == 4: # Conv Kernel [H, W, Cin, Cout] -> [Cout, Cin, H, W]
                    tensor = tensor.permute(3, 2, 0, 1).contiguous()
                    print(f"    Transposed Conv Kernel {prefix}: {tensor.shape}")
                elif tensor.ndim == 2: # Dense Kernel [Cin, Cout] -> [Cout, Cin]
                    tensor = tensor.T.contiguous()
                    print(f"    Transposed Dense Kernel {prefix}: {tensor.shape}")
                layer_kernels[prefix] = tensor
            elif param_type == 'bias':
                layer_biases[prefix] = tensor
        else:
             print(f"Warning: Parameter {varname} not found in extracted param_dict. Skipping.")


    raw_weights = []
    raw_biases = []
    for prefix in ordered_prefixes:
        if prefix in layer_kernels:
            raw_weights.append(layer_kernels[prefix])
            bias = layer_biases.get(prefix, None)
            raw_biases.append(bias)
            if bias is None:
                print(f"Warning: Kernel found for {prefix} but no bias.")
        elif prefix in layer_biases:
             print(f"Warning: Bias found for {prefix} but no kernel. Bias will be ignored unless handled specifically.")
            
    return raw_weights, raw_biases
    
    
    
def calculate_parameter_sparsity(param_list):
    total_params = 0
    zero_params = 0
    for param in param_list:
        if param is not None:
            total_params += param.numel()
            zero_params += (param == 0.0).sum().item()
    return (zero_params / total_params * 100) if total_params > 0 else 0
    
def transform_weights_biases(w, max_kernel_size, linear_as_conv=False):
    if w.ndim == 1:
        w = w.unsqueeze(-1)
        return w
    
    w = w.transpose(0, 1)
    
    if linear_as_conv:
        if w.ndim == 2:
            w = w.unsqueeze(-1).unsqueeze(-1)
        w = pad_and_flatten_kernel(w, max_kernel_size)
    else:
        w = (
            pad_and_flatten_kernel(w, max_kernel_size)
            if w.ndim == 4
            else w.unsqueeze(-1)
        )
    return w
    
def generate_static_graph_attributes(weights_template, layer_layout_list, device, direction='forward', sign_mask_value=False, fmap_size_value=1):
    print("\n--- Generating Static Architectural Attributes ---")
    static_attrs = {}

    static_attrs["conv_mask"] = [w.ndim == 4 for w in weights_template]
    print(f"  Static conv_mask: {static_attrs['conv_mask']}")

    static_attrs["layer_layout"] = layer_layout_list
    print(f"  Static layer_layout: {static_attrs['layer_layout']}")

    static_mask_hidden_tensor = None
    if len(layer_layout_list) >= 2:
        num_input_nodes = layer_layout_list[0]
        num_hidden_nodes = sum(layer_layout_list[1:-1]) if len(layer_layout_list) > 2 else 0
        num_output_nodes = layer_layout_list[-1]
        try:
            static_mask_hidden_tensor = torch.cat([
                torch.zeros(num_input_nodes, dtype=torch.bool, device=device),
                torch.ones(num_hidden_nodes, dtype=torch.bool, device=device),
                torch.zeros(num_output_nodes, dtype=torch.bool, device=device)
            ]).unsqueeze(-1)
            static_attrs["mask_hidden"] = static_mask_hidden_tensor
            print(f"  Static mask_hidden shape: {static_mask_hidden_tensor.shape}")
        except Exception as e:
            print(f"  ERROR creating mask_hidden: {e}")
            static_attrs["mask_hidden"] = None
    else:
        print("  Warning: layer_layout_list has insufficient length for mask_hidden.")
        static_attrs["mask_hidden"] = None
  
    static_node2type_tensor = None
    try:
        temp_node_types = get_node_types(layer_layout_list)
        static_node2type_tensor = temp_node_types.to(device)
        static_attrs["node2type"] = static_node2type_tensor
        print(f"  Static node2type shape: {static_node2type_tensor.shape}")
    except NameError:
         print("  ERROR: get_node_types function not found/imported!")
         static_attrs["node2type"] = None
    except Exception as e:
        print(f"  ERROR during get_node_types call: {e}")
        static_attrs["node2type"] = None

    static_edge2type_tensor = None
    try:

        static_edge2type_tensor = get_edge_types(layer_layout_list).to(device)
        static_attrs["edge2type"] = static_edge2type_tensor
        print(f"  Static edge2type shape: {static_edge2type_tensor.shape}")
    except NameError:
        print("  ERROR: get_edge_types function not found/imported!")
        static_attrs["edge2type"] = None
    except Exception as e:
        print(f"  ERROR during get_edge_types call: {e}")
        static_attrs["edge2type"] = None


    static_attrs["fmap_size"] = torch.tensor([fmap_size_value], device=device, dtype=torch.long)
    static_attrs["sign_mask"] = torch.tensor([sign_mask_value], device=device, dtype=torch.bool)
    static_attrs["direction"] = direction

    max_h = 0
    max_w = 0
    for w in weights_template:
        if w.ndim == 4:
            max_h = max(max_h, w.shape[2])
            max_w = max(max_w, w.shape[3])
    if max_h == 0 or max_w == 0:
        max_h, max_w = 3, 3
        print(f"  Warning: No 4D weights found in template. Using default max_kernel_size=({max_h},{max_w})")
    static_attrs["max_kernel_size"] = (max_h, max_w)
    print(f"  Static max_kernel_size: {static_attrs['max_kernel_size']}")

    if static_attrs.get("node2type") is None or static_attrs.get("edge2type") is None:
        print("ERROR: Failed to generate essential static graph attributes (node2type or edge2type).")

    return static_attrs


def evaluate_pruned_model(model, data_loader, device):
    """Evaluate the pruned model on validation data"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
    
    
class PrunedCNN(nn.Module):
    
    # def __init__(self, activation_fn=nn.ReLU, num_classes=10, dropout_rate=0.053518):
    def __init__(self, activation_fn=nn.ReLU, num_classes=10, dropout_rate=0.4567782701):
        super().__init__()
        self.activation_fn = activation_fn()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, stride=2),
            self.activation_fn,
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, stride=2),
            self.activation_fn,
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, stride=2),
            self.activation_fn,
            nn.Dropout(p=self.dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
            )
    
        self.fc = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def calculate_pruning_loss(
    # Dynamic inputs per step
    current_weights, current_biases,
    # Static objects / Config
    scalegmn_model, alpha, beta, device,
    # Helper functions
    transform_func, cnn_to_tg_data_func,
    # --- Pre-calculated Static Architectural Attributes ---
    static_conv_mask, static_direction, static_layer_layout,
    static_mask_hidden, static_node2type, static_edge2type,
    static_fmap_size, static_sign_mask,
    # --- Other needed args ---
    max_kernel_size
    ):
    """Calculates L1 loss and ScaleGMN prediction loss using pre-calculated static attributes."""

    # 1. L1 Loss (on current parameters)
    l1_loss = torch.tensor(0.0, device=device)
    l1_loss = l1_loss + sum(p.abs().sum() for p in current_weights)
    l1_loss = l1_loss + sum(p.abs().sum() for p in current_biases if p is not None)

    # 2. Preprocess current parameters
    transformed_weights = [transform_func(p.clone(), max_kernel_size) for p in current_weights]
    transformed_biases = [transform_func(p.clone(), max_kernel_size) if p is not None else None for p in current_biases]

    # 3. Generate graph DYNAMICALLY using CURRENT weights/biases but STATIC architecture attributes
    try:
        theta_graph = cnn_to_tg_data_func(
            # DYNAMIC parts
            weights=transformed_weights, biases=transformed_biases,
            # STATIC parts (passed as arguments)
            conv_mask=static_conv_mask, direction=static_direction,
            weights_mean=None, weights_std=None, biases_mean=None, biases_std=None,
            layer_layout=static_layer_layout, mask_hidden=static_mask_hidden,
            node2type=static_node2type, edge2type=static_edge2type,
            fmap_size=static_fmap_size, sign_mask=static_sign_mask
        )
    except Exception as e:
        print("\n--- Error during cnn_to_tg_data ---"); raise e

    theta_graph = theta_graph.to(device)
    
    # 4. Get ScaleGMN prediction
    scalegmn_model.eval()
    pred_acc_logits = scalegmn_model(theta_graph)
    pred_acc = torch.sigmoid(pred_acc_logits).squeeze(-1)
    alpha = float(alpha)

    # 5. Combine losses
    total_loss = alpha * l1_loss - (pred_acc)

    if not torch.isfinite(total_loss):
        print(f"\nWarning: NaN/Inf loss. L1={l1_loss.item()}, PredAcc={pred_acc.item()}")

    return total_loss, l1_loss.detach().item(), pred_acc.detach().item()
    
    
    
    

def magnitude_prune_weights(weights_list, biases_list, target_sparsity_fraction):
    """
    Performs global magnitude pruning on weights only across layers.
    NOTE: This modifies the input lists IN-PLACE for efficiency if they are torch tensors.
          Pass copies if you need to preserve the originals.
          Alternatively, return copies. Let's return copies for safety here.

    Args:
        weights_list (list): List of weight tensors.
        biases_list (list): List of bias tensors (or None). Will be returned unmodified but copied.
        target_sparsity_fraction (float): The target fraction of weights to prune (e.g., 0.7 for 70%).

    Returns:
        tuple: (pruned_weights_list_copy, biases_list_copy)
    """
    if target_sparsity_fraction <= 0:
        print("  Sparsity <= 0, returning copies of original weights.")
        return [p.clone() for p in weights_list], [p.clone() if p is not None else None for p in biases_list]
    if target_sparsity_fraction >= 1:
        print("  Sparsity >= 1, returning all zero weights.")
        pruned_weights = [(torch.zeros_like(p)) for p in weights_list]
        return pruned_weights, [p.clone() if p is not None else None for p in biases_list]

    print(f"  Applying global magnitude pruning for target sparsity: {target_sparsity_fraction:.2f}")
    all_weights = []
    for p in weights_list:
        if p is not None and p.ndim > 1: # Only prune weight matrices/tensors
             all_weights.append(p.detach().abs().flatten())

    if not all_weights:
        print("  No weights found to prune.")
        return [p.clone() for p in weights_list], [p.clone() if p is not None else None for p in biases_list]

    # Concatenate all weights into a single tensor to find the threshold
    flat_weights = torch.cat(all_weights)
    k = int(target_sparsity_fraction * flat_weights.numel())

    if k >= flat_weights.numel(): # Handle edge case if k is too large
         print("  Threshold calculation results in pruning all weights.")
         threshold = torch.inf
    elif k <= 0: # Handle edge case if k is zero or negative
        threshold = 0.0
    else:
        try:
            # Move flat_weights tensor to CPU for this operation
            threshold = torch.kthvalue(flat_weights.cpu(), k).values.item()
        except RuntimeError as e:
             print(f"ERROR during torch.kthvalue on CPU: {e}")
             # Fallback or re-raise depending on desired behavior
             # Example fallback: Calculate threshold via sorting on CPU
             print("Falling back to sorting method for threshold calculation.")
             sorted_weights_cpu = torch.sort(flat_weights.cpu())[0]
             threshold = sorted_weights_cpu[k-1].item() # k-1 for 0-based index

    print(f"  Calculated threshold: {threshold:.4e}")

    # Create pruned copies
    pruned_weights_copy = []
    total_elements = 0
    zero_elements = 0
    with torch.no_grad():
        for p in weights_list:
            p_copy = p.clone()
            if p is not None and p.ndim > 1:
                mask = p_copy.abs() < threshold
                p_copy[mask] = 0.0
                total_elements += p.numel()
                zero_elements += mask.sum().item()
            pruned_weights_copy.append(p_copy)

    achieved_sparsity = (zero_elements / total_elements * 100) if total_elements > 0 else 0
    print(f"  Achieved weight sparsity: {achieved_sparsity:.2f}%")

    biases_copy = [p.clone() if p is not None else None for p in biases_list]

    return pruned_weights_copy, biases_copy


def get_cifar10_loaders(batch_size, data_root='./data', num_workers=2, grayscale=True, normalize=True, pin_memory=True, val_split=0.1):
    """
    Creates CIFAR-10 DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        data_root (str): Path to the dataset root directory.
        num_workers (int): Number of workers for data loading.
        grayscale (bool): Whether to convert images to grayscale.
        normalize (bool): Whether to normalize the images.
        pin_memory (bool): Whether to use pinned memory for DataLoader.
        val_split (float): Fraction of the training data to use for validation.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.append(transforms.ToTensor())
    if normalize:
        # Standaard normalisatie voor grayscale [-1, 1]
        mean = (0.5,) if grayscale else (0.5, 0.5, 0.5)
        std = (0.5,) if grayscale else (0.5, 0.5, 0.5)
        transform_list.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(transform_list)

    try:
        # Load the full training dataset
        full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

        # Split the training dataset into training and validation datasets
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        print("CIFAR-10 DataLoaders created successfully.")
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"ERROR creating CIFAR-10 DataLoaders: {e}")
        return None, None, None
        

def fine_tune_pruned_model(model_instance, train_loader, device, epochs, lr, val_loader=None):
    """
    Fine-tunes a given CNN model instance on CIFAR-10 with early stopping.
    Assumes model_instance is already on the correct device.

    Args:
        model_instance (nn.Module): The PyTorch model to fine-tune.
        train_loader (DataLoader): DataLoader for the training set (e.g., CIFAR-10).
        device (torch.device): The device to train on.
        epochs (int): Number of fine-tuning epochs.
        lr (float): Learning rate for the Adam optimizer.
        val_loader (DataLoader, optional): DataLoader for the validation set.

    Returns:
        nn.Module: The fine-tuned model instance.
    """
    model_instance.to(device)
    model_instance.train()  # Set to train mode

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr)
    print(f"  Starting fine-tuning with Adam (lr={lr}) for {epochs} epochs...")

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(1000):
        running_loss = 0.0
        # Using tqdm for progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_instance(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"    Fine-tune Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # Validation loss for early stopping
        if val_loader is not None:
            model_instance.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model_instance(val_inputs)
                    val_loss += criterion(val_outputs, val_labels).item()
            val_loss /= len(val_loader)
            print(f"    Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Early stopping patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("    Early stopping triggered.")
                break

            model_instance.train()  # Switch back to train mode

    model_instance.eval()  # Set back to eval mode after training
    print("  Fine-tuning complete.")
    return model_instance


def get_scalegmn_prediction(weights_list_cpu, biases_list_cpu, scalegmn_model,
                             transform_func, cnn_to_tg_data_func,
                             static_graph_attrs, device):
    """
    Generates graph data and gets predicted accuracy from ScaleGMN.

    Args:
        weights_list_cpu (list): List of weight tensors (on CPU).
        biases_list_cpu (list): List of bias tensors/None (on CPU).
        scalegmn_model (nn.Module): The pre-loaded ScaleGMN model (on device).
        transform_func (callable): Function to transform weights/biases (like transform_weights_biases).
        cnn_to_tg_data_func (callable): Function to convert CNN params to graph Data object.
        static_graph_attrs (dict): Dictionary containing static graph attributes.
        device (torch.device): Device to run ScaleGMN prediction on.

    Returns:
        float: The predicted accuracy (scalar between 0 and 1). Returns 0.0 on error.
    """
    scalegmn_model.eval() # Ensure model is in eval mode

    # 1. Transform weights and biases (on CPU, then move graph to device)
    max_kernel_size = static_graph_attrs.get("max_kernel_size", (3, 3)) # Get from attrs or use default
    try:
        transformed_weights = [transform_func(p.clone(), max_kernel_size) for p in weights_list_cpu]
        transformed_biases = [transform_func(p.clone(), max_kernel_size) if p is not None else None for p in biases_list_cpu]
    except Exception as e:
        print(f"ERROR during weight transformation for ScaleGMN input: {e}")
        return 0.0

    # 2. Generate graph data
    try:
        # Create a temporary dict using only the keys needed by cnn_to_tg_data_func from static_graph_attrs
        graph_input_static_args = {
            key: static_graph_attrs.get(key) for key in [
                "conv_mask", "direction", "layer_layout", "mask_hidden",
                "node2type", "edge2type", "fmap_size", "sign_mask"
            ] if key in static_graph_attrs # Only include keys that exist
        }
        # Add None placeholders for optional args if your function requires them
        graph_input_static_args.update({
             'weights_mean': None, 'weights_std': None, 'biases_mean': None, 'biases_std': None
        })


        theta_graph = cnn_to_tg_data_func(
            weights=transformed_weights,
            biases=transformed_biases,
            **graph_input_static_args
        )
        theta_graph = theta_graph.to(device) # Move the final graph object to device
    except Exception as e:
        print(f"ERROR during graph creation (cnn_to_tg_data): {e}")
        return 0.0

    # 3. Get ScaleGMN prediction
    try:
        with torch.no_grad():
            pred_acc_logits = scalegmn_model(theta_graph)
            pred_acc = torch.sigmoid(pred_acc_logits).squeeze().item() # Get scalar value
        return pred_acc
    except Exception as e:
        print(f"ERROR during ScaleGMN prediction: {e}")
        return 0.0

# ----- End of Helper Functions -----

