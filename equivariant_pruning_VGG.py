import torch
import yaml
import numpy as np
import os
from src.utils.setup_arg_parser import setup_arg_parser
from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.models import ScaleGMN
from torch.utils.data import random_split, DataLoader
from src.utils.optim import setup_optimization
from torch.nn import CrossEntropyLoss as criterion
from src.utils.helpers import overwrite_conf, set_seed
from src.utils.CNN_utils import transform_weights_biases, mark_input_nodes, cnn_to_tg_data, generate_static_graph_attributes
from src.utils.VGG_helpers import DifferentiableVGG, load_vgg19, vgg19
import torchvision.transforms as transforms
import torchvision
import copy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
CE = criterion()

def convert_to_graph(model, conf):
    # Extract weights and biases
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data)
        elif 'bias' in name:
            biases.append(param.data)

    # Define max kernel size
    max_kernel_size = (conf['scalegmn_args']['_max_kernel_height'], conf['scalegmn_args']['_max_kernel_width'])

    # Transform weights and biases
    transformed_weights = [transform_weights_biases(w, max_kernel_size) for w in weights]
    transformed_biases = [transform_weights_biases(b, max_kernel_size) for b in biases]

    # Generate static graph attributes
    static_graph_attrs = generate_static_graph_attributes(
        weights_template=weights, 
        layer_layout_list=conf['scalegmn_args']['layer_layout'],  # VGG19 layout
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    base_graph = cnn_to_tg_data(
        weights=transformed_weights,
        biases=transformed_biases,
        conv_mask=static_graph_attrs["conv_mask"],
        direction=static_graph_attrs["direction"],
        layer_layout=static_graph_attrs["layer_layout"],
        mask_hidden=static_graph_attrs["mask_hidden"],
        node2type=static_graph_attrs["node2type"],
        edge2type=static_graph_attrs["edge2type"],
        fmap_size=static_graph_attrs["fmap_size"],
        max_kernel_size=static_graph_attrs["max_kernel_size"]
    )
    return base_graph
    
def main(args=None):
    # Read config file
    conf = yaml.safe_load(open(args.conf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device early
    set_seed(conf['train_args']['seed'])
    print("Configuration:")
    print(yaml.dump(conf, default_flow_style=False))

    output_path = conf['train_args']['output_path']
    os.makedirs(output_path, exist_ok=True)
    # =============================================================================================
    # SETUP DATASET AND DATALOADER
    # =============================================================================================
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    val_fraction = conf['data']['val_fraction']
    full_trainset = torchvision.datasets.CIFAR10(
        root=conf['data']['dataset_path'],
        train=True,
        download=True,
        transform=transform
    )

    total_train_size = len(full_trainset)
    val_size = int(total_train_size * val_fraction)
    train_size = total_train_size - val_size
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=conf['batch_size'], shuffle=True, num_workers=16)
    val_loader = DataLoader(val_subset, batch_size=conf['batch_size'], shuffle=False, num_workers=16)

    # Instantiate and move model to device
    net = ScaleGMN(conf['scalegmn_args'])
    net.to(device) # Move model to the target device

    # =============================================================================================
    # DEFINE OPTIMIZATION
    # =============================================================================================
    l1_lambda = conf['train_args']['l1_lambda']
    conf_opt = conf['optimization']
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'], optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])

    vgg = load_vgg19()
    base_graph = convert_to_graph(vgg, conf)
    first_layer_nodes = mark_input_nodes(base_graph)
    base_graph.mask_first_layer = first_layer_nodes
    base_graph.to(device)

    # =============================================================================================
    # TRAINING LOOP (Using the static theta_graph)
    # =============================================================================================
    print("Starting training loop...")
    CNN = DifferentiableVGG().to(device)

    best_val_loss = float('inf')
    best_val_ce_loss = float('inf')
    best_checkpoint_data = None
    loss_spike_threshold = conf['train_args']['loss_spike_threshold']

    for epoch in range(conf['train_args']['num_epochs']):
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            gnn_input_graph = base_graph.clone().to(device)
            vertex_features, edge_features = net(gnn_input_graph)
            vertex_features += base_graph.x
            edge_features += base_graph.edge_attr

            outputs_cnn = CNN(inputs, vertex_features, edge_features)
            ce_loss = CE(outputs_cnn, targets)
            l1_loss = CNN.sum_abs_params(vertex_features, edge_features)
            total_loss = ce_loss + l1_lambda * l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            if conf['optimization']['clip_grad']:
                torch.nn.utils.clip_grad_norm_(net.parameters(), conf['optimization']['clip_grad_max_norm'])
        
            optimizer.step()
            scheduler[0].step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Parameter sum: {l1_loss.item():.4f}, "
                      f"CE loss: {ce_loss.item():.4f}, "
                      f"Total loss: {total_loss.item():.4f}")

        # ===== Validation =====
        net.eval()
        correct = 0
        total = 0
        val_ce_loss = 0.0

        with torch.no_grad():
            gnn_input_graph = base_graph.clone().to(device)
            vertex_features, edge_features = net(gnn_input_graph)
            vertex_features += base_graph.x
            edge_features += base_graph.edge_attr
            val_l1_loss = CNN.sum_abs_params(vertex_features, edge_features)

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = CNN(inputs, vertex_features, edge_features)
                ce_loss = CE(outputs, targets)
                val_ce_loss += ce_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        val_avg_ce_loss = val_ce_loss / total
        val_avg_total_loss = val_avg_ce_loss + l1_lambda * val_l1_loss.item()
        val_accuracy = 100.0 * correct / total

        print(f"Validation Accuracy: {val_accuracy:.2f}%, "
              f"Val CE Loss: {val_avg_ce_loss:.4f}, "
              f"Total Loss: {val_avg_total_loss:.4f}")

        # ===== Check for loss spike =====
        if best_checkpoint_data is not None and val_avg_ce_loss > loss_spike_threshold:
            print(f"[!] CE loss spike detected (Current: {val_avg_ce_loss:.4f}, Best: {best_val_ce_loss:.4f}).")
            print("→ Rolling back to previous best model and halving learning rate.")
            net.load_state_dict(best_checkpoint_data['gnn_state_dict'])
            optimizer.load_state_dict(best_checkpoint_data['optimizer_state_dict'])

            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"→ New LR after rollback: {param_group['lr']:.6f}")

            net.eval()
            debug_gnn_input_graph = base_graph.clone().to(device)
            with torch.no_grad():
                debug_vf, debug_ef = net(debug_gnn_input_graph)
                debug_vf += base_graph.x
                debug_ef += base_graph.edge_attr
                debug_param_sum_eval_mode = CNN.sum_abs_params(debug_vf, debug_ef).item()
            print(f"DEBUG: Parameter sum from restored GNN in EVAL mode: {debug_param_sum_eval_mode:.4f}")
            continue

        # ===== Save best model if loss improved =====
        if val_avg_total_loss < best_val_loss:
            best_val_loss = val_avg_total_loss
            best_val_ce_loss = val_avg_ce_loss
            best_checkpoint_data = {
                'epoch': epoch,
                'gnn_state_dict': copy.deepcopy(net.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'vertex_features': vertex_features.cpu(),
                'edge_features': edge_features.cpu(),
            }
            print(f"✓ New best model saved at epoch {epoch} with val loss {best_val_loss:.4f}")

    # =============================================================================================
    # SAVE BEST CHECKPOINT AFTER TRAINING
    # =============================================================================================
    if best_checkpoint_data is not None:
        torch.save(best_checkpoint_data['gnn_state_dict'], os.path.join(output_path, 'VGG_GNN_best_parameters.pt'))
        torch.save(best_checkpoint_data['vertex_features'], os.path.join(output_path, 'VGG_vertex_features.pt'))
        torch.save(best_checkpoint_data['edge_features'], os.path.join(output_path, 'VGG_edge_features.pt'))
        print(f"✓ Best model and features saved to {output_path}")
    else:
        print("⚠ No best model found — nothing was saved.")

if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)