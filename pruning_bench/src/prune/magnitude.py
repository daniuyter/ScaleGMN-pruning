import torch

# Possibly needs to be improved to make it more efficient for large models/vgg19
def magnitude_prune(model, prune_fraction, prune_biases=False):
    """
    Zero out the smallest-magnitude weights in the model.
    prune_fraction: fraction of total weights to prune (0 <= prune_fraction <= 1)
    """
    # Gather all absolute weights into one vector
    # Gather all absolute weights into one vector
    params_to_prune = []
    for param in model.parameters():
        if param.requires_grad:
            if prune_biases or param.ndim > 1: # Weights are typically >1D, biases are 1D
                params_to_prune.append(param)
    
    if not params_to_prune:
        return model

    all_weights = torch.cat([
        p.data.abs().flatten()
        for p in params_to_prune
    ])
    # Number of weights to prune
    k = int(prune_fraction * all_weights.numel())
    if k == 0:
        return model
    # Find the k-th smallest absolute weight
    threshold, _ = torch.kthvalue(all_weights, k)
    # Zero out all weights below the threshold
    with torch.no_grad():
        for param in model.parameters():
            mask = param.data.abs() >= threshold
            param.data.mul_(mask)
    return model