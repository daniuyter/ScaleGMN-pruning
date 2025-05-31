import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from collections import OrderedDict
import math
import torch.nn as nn

class DifferentiableVGG(nn.Module):
    """
    A VGG19 architecture where weights and biases are provided externally
    in the forward pass, allowing for differentiable parameter setting.
    This implementation is based on the VGG19 structure with adaptive pooling
    and configurable fully connected layers, similar to the provided DifferentiableVGG.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_configs = [
            {'name': 'conv1_1', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv1_2', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv2_1', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv2_2', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv3_1', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv3_2', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv3_3', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv3_4', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv4_1', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv4_2', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv4_3', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv4_4', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv5_1', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv5_2', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv5_3', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv5_4', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True}
        ]

        self.adaptive_pool_output_size = (1, 1)
        last_conv_out_channels = self.conv_configs[-1]['out_channels']
        fc_in_dim = last_conv_out_channels * self.adaptive_pool_output_size[0] * self.adaptive_pool_output_size[1]

        self.fc_configs = [
            {'name': 'fc1', 'in_features': fc_in_dim, 'out_features': 512, 'dropout': 0.5},
            {'name': 'fc2', 'in_features': 512, 'out_features': 512, 'dropout': 0.5},
            {'name': 'fc3', 'in_features': 512, 'out_features': num_classes}
        ]

        self.pool_kernel_size = 2
        self.pool_stride = 2

    def forward(self, x: torch.Tensor, node_features: torch.Tensor, edge_features: torch.Tensor, bias_start_offset: int = 3) -> torch.Tensor:
        current_bias_offset = bias_start_offset
        current_edge_idx = 0

        # --- Convolutional Layers ---
        for config in self.conv_configs:
            # Parse Bias
            b_conv = node_features[current_bias_offset : current_bias_offset + config['out_channels'], 0]
            current_bias_offset += config['out_channels']

            # Parse Weights
            kernel_h, kernel_w = config['kernel_size'], config['kernel_size']
            kernel_flat_dim = kernel_h * kernel_w
            num_kernels = config['in_channels'] * config['out_channels']

            kernels_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels, :kernel_flat_dim]
            w_conv_reshaped = kernels_flat.reshape(config['in_channels'], config['out_channels'], kernel_h, kernel_w)
            w_conv = w_conv_reshaped.permute(1, 0, 2, 3)
            current_edge_idx += num_kernels

            # Apply Convolution
            x = F.conv2d(x, w_conv, b_conv, stride=config['stride'], padding=config['padding'])
            x = F.relu(x)

            if config.get('pool', False):
                x = F.max_pool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride)

        # --- Adaptive Average Pooling ---
        x = F.adaptive_avg_pool2d(x, self.adaptive_pool_output_size)
        x = torch.flatten(x, 1)

        # --- Fully Connected Layers ---
        # The remaining edge_features are for FC layers, taking only the first column value.
        fc_weights_start_row_in_edge_features = current_edge_idx
        current_fc_weight_flat_offset = 0

        for i, config in enumerate(self.fc_configs):
            # Parse Bias
            b_fc = node_features[current_bias_offset : current_bias_offset + config['out_features'], 0]
            current_bias_offset += config['out_features']

            # Parse Weights
            num_fc_weights = config['in_features'] * config['out_features']
            weights_flat = edge_features[
                fc_weights_start_row_in_edge_features + current_fc_weight_flat_offset :
                fc_weights_start_row_in_edge_features + current_fc_weight_flat_offset + num_fc_weights, 0
            ]
            w_fc_reshaped = weights_flat.reshape(config['in_features'], config['out_features'])
            w_fc = w_fc_reshaped.permute(1, 0) # PyTorch expects (out_features, in_features)
            current_fc_weight_flat_offset += num_fc_weights

            # Apply Dropout (if applicable)
            if 'dropout' in config and config['dropout'] > 0 and self.training:
                x = F.dropout(x, p=config['dropout'])

            # Apply Linear Layer
            x = F.linear(x, w_fc, b_fc)

            # Apply ReLU (if not the last layer)
            if i < len(self.fc_configs) - 1:
                x = F.relu(x)
        return x
    
    def sum_abs_params(self, node_features: torch.Tensor, edge_features: torch.Tensor, bias_start_offset: int = 3) -> torch.Tensor:
        """Returns the absolute sum of all weights and biases used in the network."""
        current_bias_offset = bias_start_offset
        current_edge_idx = 0
        total_abs_sum = 0.0

        # --- Convolutional Layers ---
        for config in self.conv_configs:
            # Bias
            b_conv = node_features[current_bias_offset : current_bias_offset + config['out_channels'], 0]
            total_abs_sum += b_conv.abs().sum()
            current_bias_offset += config['out_channels']

            # Weights
            kernel_h, kernel_w = config['kernel_size'], config['kernel_size']
            kernel_flat_dim = kernel_h * kernel_w
            num_kernels = config['in_channels'] * config['out_channels']

            kernels_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels, :kernel_flat_dim]
            total_abs_sum += kernels_flat.abs().sum()
            current_edge_idx += num_kernels

        # --- Fully Connected Layers ---
        fc_weights_start_row_in_edge_features = current_edge_idx
        current_fc_weight_flat_offset = 0

        for config in self.fc_configs:
            # Bias
            b_fc = node_features[current_bias_offset : current_bias_offset + config['out_features'], 0]
            total_abs_sum += b_fc.abs().sum()
            current_bias_offset += config['out_features']

            # Weights
            num_fc_weights = config['in_features'] * config['out_features']
            weights_flat = edge_features[
                fc_weights_start_row_in_edge_features + current_fc_weight_flat_offset :
                fc_weights_start_row_in_edge_features + current_fc_weight_flat_offset + num_fc_weights, 0
            ]
            total_abs_sum += weights_flat.abs().sum()
            current_fc_weight_flat_offset += num_fc_weights

        return total_abs_sum

def load_vgg19():
    model = vgg19()
    relative_path = 'src/data/model_best_cpu.pth.tar'
    script_dir = os.getcwd()
    checkpoint_path = os.path.abspath(os.path.join(script_dir, relative_path))
    # Load the model
    checkpoint = torch.load(checkpoint_path)
    original_state_dict = checkpoint['state_dict']

    # Create a new state_dict without the 'module.' in it (e.g. features.module.0.weight -> features.0.weight)
    new_state_dict = OrderedDict()
    for k, v in original_state_dict.items():
        name = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")
    # Print the model
    print(model)
    return model


'''
Code from https://github.com/chengyangfu/pytorch-vgg-cifar10
'''
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

class StandardVGG(nn.Module):
    def __init__(self, num_classes=10, node_features_init=None, edge_features_init=None, bias_start_offset_init=3):
        super().__init__()

        self.conv_configs = [
            {'name': 'conv1_1', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv1_2', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv2_1', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv2_2', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv3_1', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv3_2', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv3_3', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv3_4', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv4_1', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv4_2', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv4_3', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv4_4', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True},
            {'name': 'conv5_1', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv5_2', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv5_3', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'name': 'conv5_4', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': True}
        ]
        self.pool_kernel_size = 2
        self.pool_stride = 2
        
        self.features = self._make_conv_layers(self.conv_configs)

        self.adaptive_pool_output_size = (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(self.adaptive_pool_output_size)

        last_conv_out_channels = self.conv_configs[-1]['out_channels']
        fc_in_dim = last_conv_out_channels * self.adaptive_pool_output_size[0] * self.adaptive_pool_output_size[1]

        self.fc_configs = [
            {'name': 'fc1', 'in_features': fc_in_dim, 'out_features': 512, 'dropout': 0.5},
            {'name': 'fc2', 'in_features': 512, 'out_features': 512, 'dropout': 0.5},
            {'name': 'fc3', 'in_features': 512, 'out_features': num_classes}
        ]
        self.classifier = self._make_fc_layers(self.fc_configs)

        if node_features_init is not None and edge_features_init is not None:
            self._initialize_parameters_from_features(node_features_init, edge_features_init, bias_start_offset_init)
        else:
            self._initialize_default_weights() # Standard PyTorch initializations

    def _make_conv_layers(self, conv_configs):
        layers = []
        for config in conv_configs:
            conv2d = nn.Conv2d(
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                padding=config['padding']
            )
            layers.append(conv2d)
            layers.append(nn.ReLU(inplace=True))
            if config.get('pool', False):
                layers.append(nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))
        return nn.Sequential(*layers)

    def _make_fc_layers(self, fc_configs):
        layers = []
        for i, config in enumerate(fc_configs):
            if 'dropout' in config and config['dropout'] > 0:
                layers.append(nn.Dropout(p=config['dropout']))
            
            linear_layer = nn.Linear(
                in_features=config['in_features'],
                out_features=config['out_features']
            )
            layers.append(linear_layer)

            if i < len(fc_configs) - 1: # Apply ReLU to all but the last FC layer
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _initialize_default_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # VGG often uses this for FC layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _initialize_parameters_from_features(self, node_features, edge_features, bias_start_offset):
        current_bias_offset = bias_start_offset
        current_edge_idx = 0

        # Initialize Convolutional Layers
        conv_layers_iter = (layer for layer in self.features if isinstance(layer, nn.Conv2d))
        for config in self.conv_configs:
            actual_conv_layer = next(conv_layers_iter)

            # Parse Bias for conv
            b_conv_data = node_features[current_bias_offset : current_bias_offset + config['out_channels'], 0]
            current_bias_offset += config['out_channels']

            # Parse Weights for conv
            kernel_h, kernel_w = config['kernel_size'], config['kernel_size']
            kernel_flat_dim = kernel_h * kernel_w
            num_kernels = config['in_channels'] * config['out_channels']

            kernels_flat = edge_features[current_edge_idx : current_edge_idx + num_kernels, :kernel_flat_dim]
            w_conv_reshaped = kernels_flat.reshape(config['in_channels'], config['out_channels'], kernel_h, kernel_w)
            w_conv_data = w_conv_reshaped.permute(1, 0, 2, 3) # (out_channels, in_channels, H, W)
            current_edge_idx += num_kernels
            
            if actual_conv_layer.weight.shape != w_conv_data.shape:
                raise ValueError(f"Shape mismatch for conv weight {config['name']}: layer expected {actual_conv_layer.weight.shape}, features provided {w_conv_data.shape}")
            if actual_conv_layer.bias.shape != b_conv_data.shape:
                raise ValueError(f"Shape mismatch for conv bias {config['name']}: layer expected {actual_conv_layer.bias.shape}, features provided {b_conv_data.shape}")

            actual_conv_layer.weight.data = w_conv_data.clone().detach().requires_grad_(True)
            actual_conv_layer.bias.data = b_conv_data.clone().detach().requires_grad_(True)

        # Initialize Fully Connected Layers
        fc_weights_start_row_in_edge_features = current_edge_idx
        current_fc_weight_flat_offset = 0

        fc_layers_iter = (layer for layer in self.classifier if isinstance(layer, nn.Linear))
        for config in self.fc_configs:
            actual_fc_layer = next(fc_layers_iter)

            # Parse Bias for FC
            b_fc_data = node_features[current_bias_offset : current_bias_offset + config['out_features'], 0]
            current_bias_offset += config['out_features']

            # Parse Weights for FC
            num_fc_weights = config['in_features'] * config['out_features']
            
            weights_flat = edge_features[
                fc_weights_start_row_in_edge_features + current_fc_weight_flat_offset :
                fc_weights_start_row_in_edge_features + current_fc_weight_flat_offset + num_fc_weights, 0
            ]
            # Reshape to (in_features, out_features) then permute to (out_features, in_features)
            # This matches nn.Linear's expected weight shape.
            w_fc_data = weights_flat.reshape(config['in_features'], config['out_features']).permute(1, 0)
            current_fc_weight_flat_offset += num_fc_weights

            if actual_fc_layer.weight.shape != w_fc_data.shape:
                 raise ValueError(f"Shape mismatch for FC weight {config['name']}: layer expected {actual_fc_layer.weight.shape}, features provided {w_fc_data.shape}")
            if actual_fc_layer.bias.shape != b_fc_data.shape:
                raise ValueError(f"Shape mismatch for FC bias {config['name']}: layer expected {actual_fc_layer.bias.shape}, features provided {b_fc_data.shape}")

            actual_fc_layer.weight.data = w_fc_data.clone().detach().requires_grad_(True)
            actual_fc_layer.bias.data = b_fc_data.clone().detach().requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def prune_by_magnitude(self, sparsity: float):
        """
        Sets a percentage of parameters to 0 based on global magnitude pruning.
        This version is deterministic and ensures the exact sparsity level.
        
        Args:
            sparsity (float): Fraction of parameters to set to zero (e.g., 0.8 = 80%)
        """
        assert 0 <= sparsity < 1, "Sparsity must be in [0, 1)."

        # Collect references to the parameters to be pruned
        params_to_prune = []
        for name, param in self.named_parameters():
            # We only prune layers that require gradients (e.g., weights, not biases typically)
            if param.requires_grad:
                params_to_prune.append(param)
        
        if not params_to_prune:
            return # Nothing to prune

        # 1. Concatenate all parameters into a single flat tensor
        all_params_flat = torch.cat([p.detach().flatten() for p in params_to_prune])

        # Calculate the number of weights to prune
        k = int(all_params_flat.numel() * sparsity)

        if k == 0:
            return  # Nothing to prune

        # 2. Find the indices of the k smallest weights
        # We sort the absolute values and take the first k indices.
        # `stable=True` ensures that for equal-valued weights, their original
        # relative order is maintained, making the sort fully deterministic.
        _, indices_to_prune = torch.sort(all_params_flat.abs(), stable=True)
        prune_indices = indices_to_prune[:k]

        # 3. Create a single, flat mask and set the prune indices to zero
        mask_flat = torch.ones_like(all_params_flat)
        mask_flat[prune_indices] = 0.0

        # 4. Apply the mask by splitting it and reshaping it for each parameter tensor
        with torch.no_grad():
            start_idx = 0
            for param in params_to_prune:
                num_param_elements = param.numel()
                end_idx = start_idx + num_param_elements
                
                # Reshape the corresponding part of the flat mask to match the parameter's shape
                param_mask = mask_flat[start_idx:end_idx].view_as(param)
                
                # Apply the mask
                param *= param_mask

                start_idx = end_idx

    def count_zero_kernels(self):
        """Counts the number of kernels that are all zeros in convolutional layers
        and reports it as a percentage of the total number of kernels.
        Returns:
            tuple: (zero_kernel_count, percentage_zero_kernels, total_kernels_count)
        """
        zero_kernel_count = 0
        total_kernels_count = 0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # layer.weight is (out_channels, in_channels, H, W)
                # Each individual kernel is of shape (H, W)
                # Total kernels in this layer = out_channels * in_channels
                num_kernels_in_layer = layer.out_channels * layer.in_channels
                total_kernels_count += num_kernels_in_layer

                # Reshape to (out_channels * in_channels, H, W) to iterate over each kernel
                # Use .reshape() instead of .view() to avoid stride issues
                kernels = layer.weight.reshape(num_kernels_in_layer, layer.kernel_size[0], layer.kernel_size[1])
                
                # Count non-zero elements per kernel
                non_zero_counts_per_kernel = torch.count_nonzero(kernels, dim=(1, 2))
                
                # Sum up kernels that have zero non-zero elements
                zero_kernel_count += (non_zero_counts_per_kernel == 0).sum().item()
        
        if total_kernels_count == 0:
            percentage_zero_kernels = 0.0 # Avoid division by zero if no conv layers
        else:
            percentage_zero_kernels = (zero_kernel_count / total_kernels_count) * 100
            
        return zero_kernel_count, percentage_zero_kernels, total_kernels_count
    
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

    def sum_abs_params(self):
        total = 0.0
        for param in self.parameters():
            total += param.abs().sum()
        return total