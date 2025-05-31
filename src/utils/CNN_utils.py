import torch
from torch_geometric.data import Data
import torch.nn.functional as F

def pad_and_flatten_kernel(kernel, max_kernel_size):
    full_padding = (
        max_kernel_size[0] - kernel.shape[2],
        max_kernel_size[1] - kernel.shape[3],
    )
    padding = (
        full_padding[0] // 2,
        full_padding[0] - full_padding[0] // 2,
        full_padding[1] // 2,
        full_padding[1] - full_padding[1] // 2,
    )
    return F.pad(kernel, padding).flatten(2, 3)

def cnn_to_graph(
        weights,
        biases,
        weights_mean=None,
        weights_std=None,
        biases_mean=None,
        biases_std=None,
):
    weights_mean = weights_mean if weights_mean is not None else [0.0] * len(weights)
    weights_std = weights_std if weights_std is not None else [1.0] * len(weights)
    biases_mean = biases_mean if biases_mean is not None else [0.0] * len(biases)
    biases_std = biases_std if biases_std is not None else [1.0] * len(biases)

    # The graph will have as many nodes as the total number of channels in the
    # CNN, plus the number of output dimensions for each linear layer
    device = weights[0].device
    num_input_nodes = weights[0].shape[0]
    num_nodes = num_input_nodes + sum(b.shape[0] for b in biases)

    edge_features = torch.zeros(
        num_nodes, num_nodes, weights[0].shape[-1], device=device
    )

    edge_feature_masks = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)

    row_offset = 0
    col_offset = num_input_nodes  # no edge to input nodes
    for i, w in enumerate(weights):
        num_in, num_out = w.shape[:2]
        edge_features[
        row_offset:row_offset + num_in, col_offset:col_offset + num_out, :w.shape[-1]
        ] = (w - weights_mean[i]) / weights_std[i]
        edge_feature_masks[row_offset:row_offset + num_in, col_offset:col_offset + num_out] = w.shape[-1] == 1
        adjacency_matrix[row_offset:row_offset + num_in, col_offset:col_offset + num_out] = True
        row_offset += num_in
        col_offset += num_out

    node_features = torch.cat(
        [
            torch.zeros((num_input_nodes, 1), device=device, dtype=biases[0].dtype),
            *[(b - biases_mean[i]) / biases_std[i] for i, b in enumerate(biases)]
        ]
    )

    return node_features, edge_features, edge_feature_masks, adjacency_matrix

def cnn_to_tg_data(
        weights,
        biases,
        conv_mask,
        direction,
        weights_mean=None,
        weights_std=None,
        biases_mean=None,
        biases_std=None,
        **kwargs,
):
    node_features, edge_features, edge_feature_masks, adjacency_matrix = cnn_to_graph(
        weights, biases, weights_mean, weights_std, biases_mean, biases_std)
    edge_index = adjacency_matrix.nonzero().t()

    num_input_nodes = weights[0].shape[0]
    cnn_sizes = [w.shape[1] for i, w in enumerate(weights) if conv_mask[i]]
    num_cnn_nodes = num_input_nodes + sum(cnn_sizes)
    send_nodes = num_input_nodes + sum(cnn_sizes[:-1])
    spatial_embed_mask = torch.zeros_like(node_features[:, 0], dtype=torch.bool)
    spatial_embed_mask[send_nodes:num_cnn_nodes] = True
    node_types = torch.cat([
        torch.zeros(num_cnn_nodes, dtype=torch.long),
        torch.ones(node_features.shape[0] - num_cnn_nodes, dtype=torch.long)
    ])

    if direction == 'forward':
        data = Data(
            x=node_features,
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            edge_index=edge_index,
            mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
            spatial_embed_mask=spatial_embed_mask,
            node_types=node_types,
            conv_mask=conv_mask,
            **kwargs,
        )
    else:
        data = Data(
            x=node_features,
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            edge_index=edge_index,
            bw_edge_index=torch.flip(edge_index, [0]),
            bw_edge_attr=torch.reciprocal(edge_features[edge_index[0], edge_index[1]]),
            mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
            spatial_embed_mask=spatial_embed_mask,
            node_types=node_types,
            conv_mask=conv_mask,
            **kwargs,
        )

    return data

def get_node_types(nodes_per_layer):
    node_types = []
    type = 0
    for i, el in enumerate(nodes_per_layer):
        if i == 0:  # first layer
            for _ in range(el):
                node_types.append(type)
                type += 1
        elif i > 0 and i < len(nodes_per_layer) - 1:  #  hidden layers
            for _ in range(el):
                node_types.append(type)
            type += 1
        elif i == len(nodes_per_layer) - 1:  # last layer
            for _ in range(el):
                node_types.append(type)
                type += 1
    return torch.tensor(node_types)


def get_edge_types(nodes_per_layer):
    edge_types = []
    type = 0
    for i, el in enumerate(nodes_per_layer[:-1]):
        if i == 0:  # first layer
            for _ in range(el):
                for neighbour in range(nodes_per_layer[i+1]):
                    edge_types.append(type)
                type += 1
        elif i > 0 and i < len(nodes_per_layer) - 2:  #  hidden layers
            for _ in range(el):
                for neighbour in range(nodes_per_layer[i+1]):
                    edge_types.append(type)
            type += 1
        elif i == len(nodes_per_layer) - 2:  # last layer
            for neighbour in range(nodes_per_layer[i + 1]):
                for _ in range(el):
                    edge_types.append(type)
                type += 1

    return torch.tensor(edge_types)

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

def mark_input_nodes(self) -> torch.Tensor:
        input_nodes = torch.tensor(
            [True for _ in range(self.layer_layout[0])] +
            [False for _ in range(sum(self.layer_layout[1:]))]).unsqueeze(-1)
        return input_nodes