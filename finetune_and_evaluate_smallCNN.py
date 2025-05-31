import torch
import yaml
import os
from src.utils.setup_arg_parser import setup_arg_parser
from src.utils.helpers import set_seed
from torch.utils.data import random_split, DataLoader
from torch.nn import CrossEntropyLoss as criterion
from src.utils.smallCNN_helpers import FixedParameterCNN
import torchvision.transforms as transforms
import torchvision
from src.utils.optim import setup_optimization
import torch.nn as nn

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
CE = criterion()

def main(args=None):
    # Read config file
    conf = yaml.safe_load(open(args.conf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(conf['train_args']['seed'])
    print("Configuration:")
    print(yaml.dump(conf, default_flow_style=False))

    # =============================================================================================
    # SETUP DATASET AND DATALOADER
    # =============================================================================================
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_fraction = conf['data']['val_fraction']
    full_trainset = torchvision.datasets.CIFAR10(
        root=conf['data']['dataset_path'],
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=conf['data']['dataset_path'],
        train=False,
        download=True,
        transform=transform
    )

    total_train_size = len(full_trainset)
    val_size = int(total_train_size * val_fraction)
    train_size = total_train_size - val_size
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=conf['batch_size'], shuffle=True, num_workers=16)
    val_loader = DataLoader(val_subset, batch_size=conf['batch_size'], shuffle=False, num_workers=16)
    test_loader = DataLoader(testset, batch_size=conf['batch_size'], shuffle=False, num_workers=16)

    # =============================================================================================
    # DEFINE LEARNING RATES TO SEARCH OVER
    # =============================================================================================

    vertex_features = torch.load(conf['checkpoint']['vertex_features']).to(device)
    edge_features = torch.load(conf['checkpoint']['edge_features']).to(device)

    prune_magnitude = conf['train_args']['magnitude']
    early_stop_patience = conf['train_args']['early_stop_patience']
    max_epochs = conf['train_args']['max_epochs']

    learning_rates = [0.01, 0.001, 0.0001]
    best_overall_val_loss = float('inf')
    best_model_state = None
    best_lr = None

    for lr in learning_rates:
        print(f"\n=== Trying learning rate: {lr} ===")

        # Reset model
        CNN = FixedParameterCNN(vertex_features, edge_features)
        CNN.to(device)
        pruning_masks = CNN.prune_and_create_mask(sparsity=prune_magnitude)

        # Setup optimizer
        conf_opt = conf['optimization']
        conf_opt['optimizer_args']['lr'] = lr
        nonzero_params = [p for name, p in CNN.named_parameters() if p.requires_grad and torch.count_nonzero(p).item() > 0]
        optimizer, scheduler = setup_optimization(nonzero_params, optimizer_name=conf_opt['optimizer_name'], optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            CNN.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs_cnn = CNN(inputs)
                ce_loss = CE(outputs_cnn, targets)

                optimizer.zero_grad()
                ce_loss.backward()
                # if conf['optimization']['clip_grad']:
                #     torch.nn.utils.clip_grad_norm_(nonzero_params, conf['optimization']['clip_grad_max_norm'])
                optimizer.step()
                scheduler[0].step()

                with torch.no_grad():
                    for name, param in CNN.named_parameters():
                        if name in pruning_masks:
                            param.data *= pruning_masks[name]

                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, CE loss: {ce_loss.item():.4f}")

            # Validation
            CNN.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = CNN(inputs)
                    loss = CE(outputs, targets)
                    val_loss += loss.item() * targets.size(0)

                    _, predicted = torch.max(outputs, dim=1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            val_loss /= total
            val_accuracy = 100.0 * correct / total
            print(f"LR {lr}, Epoch {epoch}: Validation Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} for learning rate {lr}.")
                    break

        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_model_state = CNN.state_dict()
            best_lr = lr

    # =============================================================================================
    # FINAL EVALUATION ON TEST SET WITH BEST MODEL
    # =============================================================================================
    print(f"\nBest learning rate based on validation loss: {best_lr}")
    CNN.load_state_dict(best_model_state)
    CNN.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = CNN(inputs)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_accuracy = 100.0 * correct / total
    print(f"Test accuracy with best LR ({best_lr}): {test_accuracy:.2f}%")

    total_params = 0
    nonzero_params = 0
    for param in CNN.parameters():
        total_params += param.numel()
        nonzero_params += torch.count_nonzero(param).item()
    nonzero_percentage = 100.0 * nonzero_params / total_params
    print(f"Nonzero parameters: {nonzero_params} / {total_params} ({nonzero_percentage:.2f}%)")

if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)