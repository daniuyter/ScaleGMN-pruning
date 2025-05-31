import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import pandas as pd
import numpy as np

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

class DifferentiableCNN(nn.Module):
    """
    A CNN architecture where weights and biases are provided externally
    in the forward pass, allowing for differentiable parameter setting.
    """
    def __init__(self):
        super().__init__()
        # Conv1: 1 input channel, 16 output channels, kernel_size=3, stride=2
        self.c1_in_channels = 1
        self.c1_out_channels = 16
        self.c1_kernel_size_hw = (3, 3)
        self.c1_stride = 2
        self.c1_padding = 0

        # Conv2: 16 input channels, 16 output channels, kernel_size=3, stride=2
        self.c2_in_channels = 16
        self.c2_out_channels = 16
        self.c2_kernel_size_hw = (3, 3)
        self.c2_stride = 2
        self.c2_padding = 0

        # Conv3: 16 input channels, 16 output channels, kernel_size=3, stride=2
        self.c3_in_channels = 16
        self.c3_out_channels = 16
        self.c3_kernel_size_hw = (3, 3)
        self.c3_stride = 2
        self.c3_padding = 0

        # Fully Connected (Linear) Layer
        self.fc_in_features = 16
        self.fc_out_features = 10

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, node_features: torch.Tensor, edge_features: torch.Tensor, input_dim_nodes_skip: int = 1) -> torch.Tensor:
        """
        Performs the forward pass using GNN-generated weights and biases.
        """

        # --- 1. Parse Biases from node_features ---
        current_bias_offset = input_dim_nodes_skip
        
        b_conv1 = node_features[current_bias_offset : current_bias_offset + self.c1_out_channels, 0]
        current_bias_offset += self.c1_out_channels

        b_conv2 = node_features[current_bias_offset : current_bias_offset + self.c2_out_channels, 0]
        current_bias_offset += self.c2_out_channels

        b_conv3 = node_features[current_bias_offset : current_bias_offset + self.c3_out_channels, 0]
        current_bias_offset += self.c3_out_channels

        b_fc = node_features[current_bias_offset : current_bias_offset + self.fc_out_features, 0]

        # --- 2. Parse Weights from edge_features ---
        current_edge_idx = 0

        # Conv1 weights: (out_channels, in_channels, kernel_h, kernel_w)
        c1_kh, c1_kw = self.c1_kernel_size_hw
        c1_kernel_flat_dim = c1_kh * c1_kw
        num_kernels_c1 = self.c1_in_channels * self.c1_out_channels
        
        kernels_c1_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c1, :c1_kernel_flat_dim]
        # Reshape to (in_channels, out_channels, kernel_h, kernel_w) based on GNN output order
        w_c1_reshaped = kernels_c1_flat.reshape(
            self.c1_in_channels, self.c1_out_channels, c1_kh, c1_kw
        )
        # Permute to PyTorch's expected (out_channels, in_channels, kernel_h, kernel_w)
        w_c1 = w_c1_reshaped.permute(1, 0, 2, 3)
        current_edge_idx += num_kernels_c1

        # Conv2 weights
        c2_kh, c2_kw = self.c2_kernel_size_hw
        c2_kernel_flat_dim = c2_kh * c2_kw
        num_kernels_c2 = self.c2_in_channels * self.c2_out_channels

        kernels_c2_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c2, :c2_kernel_flat_dim]
        w_c2_reshaped = kernels_c2_flat.reshape(
            self.c2_in_channels, self.c2_out_channels, c2_kh, c2_kw
        )
        w_c2 = w_c2_reshaped.permute(1, 0, 2, 3)
        current_edge_idx += num_kernels_c2

        # Conv3 weights
        c3_kh, c3_kw = self.c3_kernel_size_hw
        c3_kernel_flat_dim = c3_kh * c3_kw
        num_kernels_c3 = self.c3_in_channels * self.c3_out_channels
        
        kernels_c3_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c3, :c3_kernel_flat_dim]
        w_c3_reshaped = kernels_c3_flat.reshape(
            self.c3_in_channels, self.c3_out_channels, c3_kh, c3_kw
        )
        w_c3 = w_c3_reshaped.permute(1, 0, 2, 3)
        current_edge_idx += num_kernels_c3

        # FC weights: (out_features, in_features)
        num_weights_fc = self.fc_in_features * self.fc_out_features
        
        # For linear layers, each 'edge' corresponds to a single weight element.
        weights_fc_flat = edge_features[current_edge_idx : current_edge_idx + num_weights_fc, 0]
        # Reshape to (in_features, out_features) based on GNN output order
        w_fc_reshaped = weights_fc_flat.reshape(self.fc_in_features, self.fc_out_features)
        # Permute to PyTorch's expected (out_features, in_features)
        w_fc = w_fc_reshaped.permute(1, 0)
        # current_edge_idx += num_weights_fc # Not strictly needed after the last weight

        # --- 3. Perform Forward Pass using Functional API ---
        x = F.conv2d(x, w_c1, b_conv1, stride=self.c1_stride, padding=self.c1_padding)
        x = F.relu(x)
        x = F.conv2d(x, w_c2, b_conv2, stride=self.c2_stride, padding=self.c2_padding)
        x = F.relu(x)
        x = F.conv2d(x, w_c3, b_conv3, stride=self.c3_stride, padding=self.c3_padding)
        x = F.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, w_fc, b_fc)

        return x
    
    def sum_abs_params(self, node_features: torch.Tensor, edge_features: torch.Tensor, input_dim_nodes_skip: int = 1) -> torch.Tensor:
        """
        Computes the sum of absolute values of all CNN parameters (weights and biases)
        extracted from node_features and edge_features.
        """
        total_sum = 0.0

        # --- Biases from node_features ---
        current_bias_offset = input_dim_nodes_skip

        b_conv1 = node_features[current_bias_offset : current_bias_offset + self.c1_out_channels, 0]
        current_bias_offset += self.c1_out_channels
        total_sum += b_conv1.abs().sum()

        b_conv2 = node_features[current_bias_offset : current_bias_offset + self.c2_out_channels, 0]
        current_bias_offset += self.c2_out_channels
        total_sum += b_conv2.abs().sum()

        b_conv3 = node_features[current_bias_offset : current_bias_offset + self.c3_out_channels, 0]
        current_bias_offset += self.c3_out_channels
        total_sum += b_conv3.abs().sum()

        b_fc = node_features[current_bias_offset : current_bias_offset + self.fc_out_features, 0]
        total_sum += b_fc.abs().sum()

        # --- Weights from edge_features ---
        current_edge_idx = 0

        # Conv1 weights
        c1_kh, c1_kw = self.c1_kernel_size_hw
        c1_kernel_flat_dim = c1_kh * c1_kw
        num_kernels_c1 = self.c1_in_channels * self.c1_out_channels
        kernels_c1_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c1, :c1_kernel_flat_dim]
        total_sum += kernels_c1_flat.abs().sum()
        current_edge_idx += num_kernels_c1

        # Conv2 weights
        c2_kh, c2_kw = self.c2_kernel_size_hw
        c2_kernel_flat_dim = c2_kh * c2_kw
        num_kernels_c2 = self.c2_in_channels * self.c2_out_channels
        kernels_c2_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c2, :c2_kernel_flat_dim]
        total_sum += kernels_c2_flat.abs().sum()
        current_edge_idx += num_kernels_c2

        # Conv3 weights
        c3_kh, c3_kw = self.c3_kernel_size_hw
        c3_kernel_flat_dim = c3_kh * c3_kw
        num_kernels_c3 = self.c3_in_channels * self.c3_out_channels
        kernels_c3_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c3, :c3_kernel_flat_dim]
        total_sum += kernels_c3_flat.abs().sum()
        current_edge_idx += num_kernels_c3

        # FC weights
        num_weights_fc = self.fc_in_features * self.fc_out_features
        weights_fc_flat = edge_features[current_edge_idx : current_edge_idx + num_weights_fc, 0]
        total_sum += weights_fc_flat.abs().sum()

        return total_sum
    
    def save_gnn_parameters(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        save_path: str,
        input_dim_nodes_skip: int = 1
    ) -> None:
        """
        Saves GNN-generated CNN parameters to a .pt file.

        Args:
            node_features (Tensor): Node feature matrix with biases.
            edge_features (Tensor): Edge feature matrix with weights.
            save_path (str): Path to save the .pt file.
            input_dim_nodes_skip (int): Index to skip non-bias data in node features.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # === Parse biases ===
        current_bias_offset = input_dim_nodes_skip

        b_conv1 = node_features[current_bias_offset : current_bias_offset + self.c1_out_channels, 0]
        current_bias_offset += self.c1_out_channels

        b_conv2 = node_features[current_bias_offset : current_bias_offset + self.c2_out_channels, 0]
        current_bias_offset += self.c2_out_channels

        b_conv3 = node_features[current_bias_offset : current_bias_offset + self.c3_out_channels, 0]
        current_bias_offset += self.c3_out_channels

        b_fc = node_features[current_bias_offset : current_bias_offset + self.fc_out_features, 0]

        # === Parse weights ===
        current_edge_idx = 0

        # Conv1 weights
        c1_kh, c1_kw = self.c1_kernel_size_hw
        num_kernels_c1 = self.c1_in_channels * self.c1_out_channels
        kernels_c1_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c1, :c1_kh * c1_kw]
        w_c1 = kernels_c1_flat.reshape(self.c1_in_channels, self.c1_out_channels, c1_kh, c1_kw).permute(1, 0, 2, 3)
        current_edge_idx += num_kernels_c1

        # Conv2 weights
        c2_kh, c2_kw = self.c2_kernel_size_hw
        num_kernels_c2 = self.c2_in_channels * self.c2_out_channels
        kernels_c2_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c2, :c2_kh * c2_kw]
        w_c2 = kernels_c2_flat.reshape(self.c2_in_channels, self.c2_out_channels, c2_kh, c2_kw).permute(1, 0, 2, 3)
        current_edge_idx += num_kernels_c2

        # Conv3 weights
        c3_kh, c3_kw = self.c3_kernel_size_hw
        num_kernels_c3 = self.c3_in_channels * self.c3_out_channels
        kernels_c3_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels_c3, :c3_kh * c3_kw]
        w_c3 = kernels_c3_flat.reshape(self.c3_in_channels, self.c3_out_channels, c3_kh, c3_kw).permute(1, 0, 2, 3)
        current_edge_idx += num_kernels_c3

        # FC weights
        num_weights_fc = self.fc_in_features * self.fc_out_features
        weights_fc_flat = edge_features[current_edge_idx : current_edge_idx + num_weights_fc, 0]
        w_fc = weights_fc_flat.reshape(self.fc_in_features, self.fc_out_features).permute(1, 0)

        # === Save all parameters ===
        params = {
            "w_c1": w_c1,
            "b_c1": b_conv1,
            "w_c2": w_c2,
            "b_c2": b_conv2,
            "w_c3": w_c3,
            "b_c3": b_conv3,
            "w_fc": w_fc,
            "b_fc": b_fc
        }

        torch.save(params, save_path)

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

class StandardCNN(nn.Module):
    def __init__(self, vertex_features: torch.Tensor, edge_features: torch.Tensor, input_dim_nodes_skip: int = 1):
        super().__init__()

        # Layer configuration (same as DifferentiableCNN)
        self.c1_in_channels = 1
        self.c1_out_channels = 16
        self.c1_kernel_size = (3, 3)
        self.c1_stride = 2
        self.c1_padding = 0

        self.c2_in_channels = 16
        self.c2_out_channels = 16
        self.c2_kernel_size = (3, 3)
        self.c2_stride = 2
        self.c2_padding = 0

        self.c3_in_channels = 16
        self.c3_out_channels = 16
        self.c3_kernel_size = (3, 3)
        self.c3_stride = 2
        self.c3_padding = 0

        self.fc_in_features = 16
        self.fc_out_features = 10

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Parse biases
        offset = input_dim_nodes_skip
        b_conv1 = vertex_features[offset : offset + self.c1_out_channels, 0]
        offset += self.c1_out_channels
        b_conv2 = vertex_features[offset : offset + self.c2_out_channels, 0]
        offset += self.c2_out_channels
        b_conv3 = vertex_features[offset : offset + self.c3_out_channels, 0]
        offset += self.c3_out_channels
        b_fc = vertex_features[offset : offset + self.fc_out_features, 0]

        # Parse weights
        edge_offset = 0

        def extract_conv_weights(in_c, out_c, kh, kw):
            nonlocal edge_offset
            flat_dim = kh * kw
            n_kernels = in_c * out_c
            weights_flat = edge_features[edge_offset : edge_offset + n_kernels, :flat_dim]
            weights = weights_flat.reshape(in_c, out_c, kh, kw).permute(1, 0, 2, 3)
            edge_offset += n_kernels
            return weights

        def extract_fc_weights(in_f, out_f):
            nonlocal edge_offset
            flat = edge_features[edge_offset : edge_offset + in_f * out_f, 0]
            weights = flat.reshape(in_f, out_f).permute(1, 0)
            edge_offset += in_f * out_f
            return weights

        w_conv1 = extract_conv_weights(self.c1_in_channels, self.c1_out_channels, *self.c1_kernel_size)
        w_conv2 = extract_conv_weights(self.c2_in_channels, self.c2_out_channels, *self.c2_kernel_size)
        w_conv3 = extract_conv_weights(self.c3_in_channels, self.c3_out_channels, *self.c3_kernel_size)
        w_fc = extract_fc_weights(self.fc_in_features, self.fc_out_features)

        # Register actual layers with weights/biases
        self.conv1 = nn.Conv2d(self.c1_in_channels, self.c1_out_channels, self.c1_kernel_size,
                               stride=self.c1_stride, padding=self.c1_padding, bias=True)
        self.conv2 = nn.Conv2d(self.c2_in_channels, self.c2_out_channels, self.c2_kernel_size,
                               stride=self.c2_stride, padding=self.c2_padding, bias=True)
        self.conv3 = nn.Conv2d(self.c3_in_channels, self.c3_out_channels, self.c3_kernel_size,
                               stride=self.c3_stride, padding=self.c3_padding, bias=True)
        self.fc = nn.Linear(self.fc_in_features, self.fc_out_features, bias=True)

        # Assign weights
        with torch.no_grad():
            self.conv1.weight.copy_(w_conv1)
            self.conv1.bias.copy_(b_conv1)

            self.conv2.weight.copy_(w_conv2)
            self.conv2.bias.copy_(b_conv2)

            self.conv3.weight.copy_(w_conv3)
            self.conv3.bias.copy_(b_conv3)

            self.fc.weight.copy_(w_fc)
            self.fc.bias.copy_(b_fc)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def sum_abs_params(self):
        total = 0.0
        for param in self.parameters():
            total += param.abs().sum()
        return total
    
class FixedParameterCNN(nn.Module):
    def __init__(self, vertex_features: torch.Tensor, edge_features: torch.Tensor, input_dim_nodes_skip: int = 1, dropout_rate=0.053518):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Layer configuration (same as DifferentiableCNN)
        self.c1_in_channels = 1
        self.c1_out_channels = 16
        self.c1_kernel_size = (3, 3)
        self.c1_stride = 2
        self.c1_padding = 0

        self.c2_in_channels = 16
        self.c2_out_channels = 16
        self.c2_kernel_size = (3, 3)
        self.c2_stride = 2
        self.c2_padding = 0

        self.c3_in_channels = 16
        self.c3_out_channels = 16
        self.c3_kernel_size = (3, 3)
        self.c3_stride = 2
        self.c3_padding = 0

        self.fc_in_features = 16
        self.fc_out_features = 10

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Parse biases
        offset = input_dim_nodes_skip
        b_conv1 = vertex_features[offset : offset + self.c1_out_channels, 0]
        offset += self.c1_out_channels
        b_conv2 = vertex_features[offset : offset + self.c2_out_channels, 0]
        offset += self.c2_out_channels
        b_conv3 = vertex_features[offset : offset + self.c3_out_channels, 0]
        offset += self.c3_out_channels
        b_fc = vertex_features[offset : offset + self.fc_out_features, 0]

        # Parse weights
        edge_offset = 0

        def extract_conv_weights(in_c, out_c, kh, kw):
            nonlocal edge_offset
            flat_dim = kh * kw
            n_kernels = in_c * out_c
            weights_flat = edge_features[edge_offset : edge_offset + n_kernels, :flat_dim]
            weights = weights_flat.reshape(in_c, out_c, kh, kw).permute(1, 0, 2, 3)
            edge_offset += n_kernels
            return weights

        def extract_fc_weights(in_f, out_f):
            nonlocal edge_offset
            flat = edge_features[edge_offset : edge_offset + in_f * out_f, 0]
            weights = flat.reshape(in_f, out_f).permute(1, 0)
            edge_offset += in_f * out_f
            return weights

        w_conv1 = extract_conv_weights(self.c1_in_channels, self.c1_out_channels, *self.c1_kernel_size)
        w_conv2 = extract_conv_weights(self.c2_in_channels, self.c2_out_channels, *self.c2_kernel_size)
        w_conv3 = extract_conv_weights(self.c3_in_channels, self.c3_out_channels, *self.c3_kernel_size)
        w_fc = extract_fc_weights(self.fc_in_features, self.fc_out_features)

        # Register actual layers with weights/biases
        self.conv1 = nn.Conv2d(self.c1_in_channels, self.c1_out_channels, self.c1_kernel_size,
                            stride=self.c1_stride, padding=self.c1_padding, bias=True)
        self.conv2 = nn.Conv2d(self.c2_in_channels, self.c2_out_channels, self.c2_kernel_size,
                            stride=self.c2_stride, padding=self.c2_padding, bias=True)
        self.conv3 = nn.Conv2d(self.c3_in_channels, self.c3_out_channels, self.c3_kernel_size,
                            stride=self.c3_stride, padding=self.c3_padding, bias=True)
        self.fc = nn.Linear(self.fc_in_features, self.fc_out_features, bias=True)

        # Assign weights
        with torch.no_grad():
            self.conv1.weight.copy_(w_conv1)
            self.conv1.bias.copy_(b_conv1)

            self.conv2.weight.copy_(w_conv2)
            self.conv2.bias.copy_(b_conv2)

            self.conv3.weight.copy_(w_conv3)
            self.conv3.bias.copy_(b_conv3)

            self.fc.weight.copy_(w_fc)
            self.fc.bias.copy_(b_fc)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def prune_and_create_mask(self, sparsity: float):
        """
        Prunes the model and returns a dictionary of masks to be enforced during training.
        """
        masks = {}
        params_to_prune = []
        param_names = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                params_to_prune.append(param)
                param_names.append(name)
        
        # --- (Your deterministic pruning logic from the previous answer) ---
        all_params_flat = torch.cat([p.detach().flatten() for p in params_to_prune])
        k = int(all_params_flat.numel() * sparsity)
        if k == 0:
            return {}

        _, indices_to_prune = torch.sort(all_params_flat.abs(), stable=True)
        prune_indices = indices_to_prune[:k]
        
        mask_flat = torch.ones_like(all_params_flat)
        mask_flat[prune_indices] = 0.0
        # --- (End of pruning logic) ---

        # Now, create and apply the masks, and store them
        with torch.no_grad():
            start_idx = 0
            for i, param in enumerate(params_to_prune):
                num_param_elements = param.numel()
                end_idx = start_idx + num_param_elements
                
                param_mask = mask_flat[start_idx:end_idx].view_as(param)
                param *= param_mask # Apply initial pruning
                
                masks[param_names[i]] = param_mask # Store the mask
                
                start_idx = end_idx
                
        return masks