import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.evaluate import evaluate_model
import copy

# Assuming evaluate_model is defined in another file (e.g., evaluate.py or utils/evaluation.py)
# and will be imported where needed. For example:
# from evaluate import evaluate_model 

# It's good practice to have calculate_parameter_sparsity as a global helper if used elsewhere
def calculate_parameter_sparsity(model_params):
   if not model_params:
       return 0.0
   total_params = 0
   zero_params = 0
   for param in model_params:
       total_params += param.numel()
       zero_params += (param.data == 0).sum().item()
   return (zero_params / total_params) * 100 if total_params > 0 else 0.0

def fine_tune_pruned_model(model_to_fine_tune, 
                           ft_train_loader,
                           ft_val_loader, # Keep for epoch-wise validation monitoring
                           max_epochs_safeguard=200, 
                           finetune_lr=1e-3, 
                           current_device=None,
                           patience=5,
                           min_delta=1e-5):
    """
    Fine-tunes a given pruned model, keeping pruned weights at zero.

    Args:
        model_to_fine_tune (torch.nn.Module): The pruned model to be fine-tuned.
        ft_train_loader (DataLoader): DataLoader for the training set.
        ft_val_loader (DataLoader): DataLoader for the validation set (for epoch monitoring).
        max_epochs_safeguard (int): Max number of epochs for fine-tuning.
        finetune_lr (float): Learning rate for the Adam optimizer.
        current_device (torch.device, optional): The device to run on. If None, uses CUDA or CPU.

    Returns:
        torch.nn.Module: The fine-tuned model.
    """
    if current_device is None:
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {current_device}")

    ft_model = model_to_fine_tune.to(current_device)

    for param in ft_model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ft_model.parameters(), lr=finetune_lr)
    print(f"Fine-tuning optimizer: {type(optimizer).__name__}, LR: {finetune_lr}")

    zero_masks = []
    n_zeros_in_mask = 0
    with torch.no_grad():
        for param in ft_model.parameters():
            if isinstance(param, torch.Tensor) and param.data is not None:
                current_mask = (param.data == 0.0)
                zero_masks.append(current_mask.to(current_device))
                n_zeros_in_mask += current_mask.sum().item()
            else:
                zero_masks.append(None) 
    print(f"Created masks for {len(zero_masks)} parameter groups. Total elements to keep zero: {n_zeros_in_mask}")

    best_val_loss = float('inf') 
    val_accuracy_at_best_loss = 0.0 # NEW: To store val acc at best val loss
    epochs_no_improve = 0
    best_model_state = None 
    epochs_run = 0 # NEW: To store number of epochs run

    print(f"--- Running Fine-tuning with Early Stopping (Patience: {patience}, Min Delta: {min_delta}, Max Epochs: {max_epochs_safeguard}) ---")

    for ft_epoch in range(max_epochs_safeguard): 
        epochs_run = ft_epoch + 1 # Update epochs run
        ft_model.train()
        running_loss = 0.0
        num_batches = 0
        for inputs, labels in tqdm(ft_train_loader, desc=f"Fine-tune Epoch {ft_epoch+1}/{max_epochs_safeguard}", leave=False):
            inputs, labels = inputs.to(current_device), labels.to(current_device)

            optimizer.zero_grad()
            outputs = ft_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Apply masks to gradients to keep pruned weights zero
            with torch.no_grad():
                for i, param in enumerate(ft_model.parameters()):
                    if param.grad is not None and i < len(zero_masks) and zero_masks[i] is not None:
                        param.grad.data[zero_masks[i]] = 0.0 # Use .data here if zero_masks is boolean
            
            optimizer.step()

            # ADD THIS: Re-apply masks to weights after optimizer step for robustness
            with torch.no_grad():
                for i, param in enumerate(ft_model.parameters()):
                    if i < len(zero_masks) and zero_masks[i] is not None:
                        # Ensure param.data is used for in-place modification
                        param.data[zero_masks[i]] = 0.0 
                        # Or, if zero_masks define what to KEEP (like in Method 2's original masks):
                        # param.data.mul_(keep_masks[i]) # where keep_masks = ~zero_masks[i]

            running_loss += loss.item()
            num_batches += 1
        
        epoch_loss = running_loss / num_batches if num_batches > 0 else 0
        
        current_val_accuracy_epoch = -1.0 
        current_val_loss_epoch = -1.0

        if ft_val_loader:
            ft_model.eval() 
            val_accuracy_epoch, current_val_loss_epoch = evaluate_model(ft_model, ft_val_loader, current_device, criterion=criterion) 

            print(f"Epoch {ft_epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {current_val_loss_epoch:.4f}, Val Acc: {val_accuracy_epoch * 100:.2f}%")
            
            if current_val_loss_epoch < best_val_loss - min_delta:
                best_val_loss = current_val_loss_epoch
                val_accuracy_at_best_loss = val_accuracy_epoch # Store val_acc at this point
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(ft_model.state_dict())
                print(f"  New best validation loss: {best_val_loss:.4f} (Val Acc: {val_accuracy_at_best_loss * 100:.2f}%)")
            else:
                epochs_no_improve += 1
                print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {ft_epoch+1} epochs (val loss not improving).")
                break 
            ft_model.train() 
        else:
            print(f"Epoch {ft_epoch+1}/{max_epochs_safeguard}, Training Loss: {epoch_loss:.4f} (No validation for early stopping)")
            # If no validation, best_val_loss remains inf, val_accuracy_at_best_loss remains 0.0
            # The model from the last epoch will be used.
            if ft_epoch == max_epochs_safeguard -1 : # If it's the last epoch and no val
                 best_model_state = copy.deepcopy(ft_model.state_dict()) # Save last state
                 # best_val_loss and val_accuracy_at_best_loss will be their defaults (inf, 0)
                 # which is fine as they won't be selected as "best" if other runs have validation.


    if best_model_state:
        print(f"Loading best model state (Val Loss: {best_val_loss:.4f}, Val Acc: {val_accuracy_at_best_loss*100:.2f}%).")
        ft_model.load_state_dict(best_model_state)
    elif not ft_val_loader and epochs_run > 0 : # No validation, but training happened
        print("Fine-tuning ran without validation, using the model from the last epoch.")
        # ft_model is already the last state, best_model_state might be None if loop didn't run max_epochs
        # and no val_loader. Ensure ft_model is what we want.
        # If epochs_run is max_epochs_safeguard, best_model_state would have been set above.
        # If loop broke early for other reasons (not possible without val_loader), this is a fallback.
    else: # No improvement or no training happened with validation
        print("No improvement in validation loss or no validation training, using model from the last epoch or initial state.")


    print("\n--- Fine-tuning complete ---")
    return ft_model, best_val_loss, val_accuracy_at_best_loss, epochs_run
