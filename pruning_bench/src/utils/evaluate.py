import torch
import torch.nn as nn # For CrossEntropyLoss

def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluates the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set.
        device (torch.device): The device to run evaluation on.
        criterion (torch.nn.Module, optional): Loss function. If None, loss is not calculated.

    Returns:
        tuple: (accuracy, average_loss)
               If criterion is None, average_loss will be 0.0.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculations
        for data in data_loader: # Iterate over the provided data_loader
            images, labels = data
            images, labels = images.to(device), labels.to(device) # Move data to the specified device

            # Grayscale to RGB conversion if model expects 3 channels and gets 1
            # This check assumes the first parameter of the model (e.g., first conv layer's weights)
            # can indicate the expected input channels. This might need adjustment for some architectures.
            if images.shape[1] == 1: # Input is grayscale
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.ndim > 1 and first_param.shape[1] == 3: # Model expects 3 channels
                    images = images.repeat(1, 3, 1, 1) # Repeat grayscale channel to simulate RGB

            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_batches += 1
    
    accuracy = correct / total if total > 0 else 0.0
    average_loss = running_loss / num_batches if num_batches > 0 and criterion else 0.0
    
    return accuracy, average_loss
