import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any
import math

class SigilLayer(nn.Module):
    """
    Enhanced wrapper for any PyTorch layer with Kintsugi-guided adaptive parameters.
    
    Features:
    - Supports Linear, Conv1d/2d/3d, and other common layers
    - Optional phi modulation in forward pass
    - Kintsugi-guided parameter updates based on loss gradients
    - Memory-efficient history tracking with configurable limits
    - Full PyTorch training loop compatibility
    
    Args:
        base_layer: The PyTorch layer to wrap
        name: Optional name for the layer
        use_phi_modulation: Whether to apply phi modulation in forward pass
        kintsugi_lr: Learning rate for Kintsugi updates (default: 1e-4)
        history_limit: Maximum number of phi states to store (default: 100)
        phi_init_std: Standard deviation for phi initialization (default: 0.01)
    """
    
    def __init__(
        self, 
        base_layer: nn.Module, 
        name: Optional[str] = None,
        use_phi_modulation: bool = True,
        kintsugi_lr: float = 1e-4,
        history_limit: int = 100,
        phi_init_std: float = 0.01
    ):
        super().__init__()
        self.base_layer = base_layer
        self.name = name or f"Sigil_{base_layer.__class__.__name__}"
        self.use_phi_modulation = use_phi_modulation
        self.kintsugi_lr = kintsugi_lr
        self.history_limit = history_limit
        
        # Determine phi dimensions based on layer type
        self.phi_shape = self._get_phi_shape(base_layer)
        
        # Initialize phi parameter
        self.phi = nn.Parameter(torch.randn(self.phi_shape) * phi_init_std + 1.0)
        
        # History tracking
        self.phi_history = []
        self.loss_history = []
        self.epoch_count = 0
        
        # Kintsugi state tracking
        self._last_loss = None
        self._phi_momentum = torch.zeros_like(self.phi.data)
        self.momentum_decay = 0.9
        self._loss_window = []  # For longer trend analysis
        
    def _get_phi_shape(self, layer: nn.Module) -> torch.Size:
        """Determine appropriate phi shape based on layer type."""
        if isinstance(layer, nn.Linear):
            return torch.Size([layer.out_features])
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return torch.Size([layer.out_channels])
        elif hasattr(layer, 'num_features'):  # BatchNorm, etc.
            return torch.Size([layer.num_features])
        elif hasattr(layer, 'out_features'):
            return torch.Size([layer.out_features])
        elif hasattr(layer, 'out_channels'):
            return torch.Size([layer.out_channels])
        else:
            # Fallback: try to infer from a dummy forward pass
            try:
                with torch.no_grad():
                    dummy_input = self._create_dummy_input(layer)
                    if dummy_input is not None:
                        dummy_output = layer(dummy_input)
                        if len(dummy_output.shape) >= 2:
                            return torch.Size([dummy_output.shape[1]])  # Channel dimension
            except:
                pass
            
            # Final fallback
            return torch.Size([64])
    
    def _create_dummy_input(self, layer: nn.Module) -> Optional[torch.Tensor]:
        """Create a dummy input tensor for shape inference."""
        if isinstance(layer, nn.Linear):
            return torch.randn(1, layer.in_features)
        elif isinstance(layer, nn.Conv1d):
            return torch.randn(1, layer.in_channels, 32)
        elif isinstance(layer, nn.Conv2d):
            return torch.randn(1, layer.in_channels, 32, 32)
        elif isinstance(layer, nn.Conv3d):
            return torch.randn(1, layer.in_channels, 16, 16, 16)
        return None
    
    def _apply_phi_modulation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply phi modulation to the output tensor."""
        if not self.use_phi_modulation:
            return x
        
        # Ensure phi is on the same device as x
        phi = self.phi.to(x.device)
        
        # Handle different tensor shapes with runtime validation
        if len(x.shape) == 2:  # Linear layer output [batch, features]
            assert x.shape[1] == phi.shape[0], f"Phi shape {phi.shape} incompatible with output shape {x.shape}"
            return x * phi.unsqueeze(0)
        elif len(x.shape) == 3:  # Conv1d output [batch, channels, length]
            assert x.shape[1] == phi.shape[0], f"Phi shape {phi.shape} incompatible with output shape {x.shape}"
            return x * phi.unsqueeze(0).unsqueeze(-1)
        elif len(x.shape) == 4:  # Conv2d output [batch, channels, height, width]
            assert x.shape[1] == phi.shape[0], f"Phi shape {phi.shape} incompatible with output shape {x.shape}"
            return x * phi.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif len(x.shape) == 5:  # Conv3d output [batch, channels, depth, height, width]
            assert x.shape[1] == phi.shape[0], f"Phi shape {phi.shape} incompatible with output shape {x.shape}"
            return x * phi.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            # Fallback: broadcast along the second dimension
            assert len(x.shape) >= 2 and x.shape[1] == phi.shape[0], \
                f"Phi shape {phi.shape} incompatible with output shape {x.shape}"
            phi_expanded = phi.view(1, -1, *([1] * (len(x.shape) - 2)))
            return x * phi_expanded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional phi modulation."""
        base_output = self.base_layer(x)
        return self._apply_phi_modulation(base_output)
    
    def update_phi(self, loss: Optional[torch.Tensor] = None, loss_map: Optional[Dict[str, Any]] = None):
        """
        Update phi using Kintsugi-guided adaptive mechanism.
        
        Args:
            loss: Current loss tensor (optional)
            loss_map: Dictionary containing loss information and gradients (optional)
        """
        if not self.training:
            return
        
        # Extract loss value
        current_loss = None
        if loss is not None:
            current_loss = loss.item() if hasattr(loss, 'item') else float(loss)
        elif loss_map and 'loss' in loss_map:
            current_loss = loss_map['loss']
        
        # Ensure momentum is on the same device
        if self._phi_momentum.device != self.phi.device:
            self._phi_momentum = self._phi_momentum.to(self.phi.device)
        with torch.no_grad():
            if self.phi.grad is not None:
                # Use actual gradients if available (preferred)
                phi_grad = self.phi.grad.data
                
                # Apply momentum
                self._phi_momentum = (self.momentum_decay * self._phi_momentum + 
                                    (1 - self.momentum_decay) * phi_grad)
                
                # Kintsugi-guided update with adaptive learning rate
                adaptive_lr = self._compute_adaptive_lr(current_loss)
                self.phi.data -= adaptive_lr * self._phi_momentum
                
            elif current_loss is not None and self._last_loss is not None:
                # Finite difference estimation for gradient-free update
                loss_delta = current_loss - self._last_loss
                
                if abs(loss_delta) > 1e-8:
                    # Use finite difference approximation with small perturbation
                    # Estimate gradient direction from recent phi changes and loss changes
                    if len(self.phi_history) >= 2:
                        phi_delta = self.phi.data - self.phi_history[-1].to(self.phi.device)
                        phi_norm = phi_delta.norm()
                        if phi_norm > 1e-8:
                            # Gradient estimate: loss_change / phi_change direction
                            estimated_grad = (loss_delta / phi_norm) * phi_delta
                        else:
                            estimated_grad = torch.sign(loss_delta) * torch.randn_like(self.phi) * 0.1
                    else:
                        estimated_grad = torch.sign(loss_delta) * torch.randn_like(self.phi) * 0.1
                    
                    self.phi.data -= self.kintsugi_lr * estimated_grad
            else:
                # Exploratory update when no gradient information is available
                noise_scale = self.kintsugi_lr * 0.1
                self.phi.data += torch.randn_like(self.phi) * noise_scale
            
            # Apply constraints to keep phi values reasonable
            self.phi.data.clamp_(0.1, 10.0)
        
        # Update loss history
        if current_loss is not None:
            self._last_loss = current_loss
    
    def _compute_adaptive_lr(self, current_loss: Optional[float]) -> float:
        """Compute adaptive learning rate based on loss trends."""
        if current_loss is None:
            return self.kintsugi_lr
        
        # Add to loss window for trend analysis
        self._loss_window.append(current_loss)
        if len(self._loss_window) > 20:  # Keep last 20 losses
            self._loss_window.pop(0)
        
        if len(self._loss_window) < 3:
            return self.kintsugi_lr
        
        # Short-term trend (last 5 losses)
        short_window = self._loss_window[-5:] if len(self._loss_window) >= 5 else self._loss_window
        short_trend = 0
        if len(short_window) >= 2:
            short_trend = sum(short_window[i] - short_window[i-1] 
                            for i in range(1, len(short_window))) / (len(short_window) - 1)
        
        # Long-term trend (using moving average)
        if len(self._loss_window) >= 10:
            mid_point = len(self._loss_window) // 2
            early_avg = sum(self._loss_window[:mid_point]) / mid_point
            recent_avg = sum(self._loss_window[mid_point:]) / (len(self._loss_window) - mid_point)
            long_trend = recent_avg - early_avg
        else:
            long_trend = short_trend
        
        # Combine trends for adaptive rate
        if short_trend < -1e-5 and long_trend < -1e-5:  # Consistently decreasing
            return self.kintsugi_lr * 1.5
        elif short_trend < -1e-5:  # Short-term decrease
            return self.kintsugi_lr * 1.2
        elif short_trend > 1e-5:  # Increasing loss
            return self.kintsugi_lr * 0.7
        else:  # Stable
            return self.kintsugi_lr
    
    def record_epoch(self, loss: Optional[float] = None):
        """Record current state at end of epoch."""
        self.epoch_count += 1
        
        # Store phi state (with memory limit)
        self.phi_history.append(self.phi.detach().cpu().clone())
        if len(self.phi_history) > self.history_limit:
            self.phi_history.pop(0)
        
        # Store loss if provided
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.history_limit:
                self.loss_history.pop(0)
    
    def get_phi(self) -> torch.Tensor:
        """Get current phi values."""
        return self.phi.detach().clone()
    
    def get_phi_stats(self) -> Dict[str, float]:
        """Get statistics about phi values."""
        phi_data = self.phi.detach()
        return {
            'mean': phi_data.mean().item(),
            'std': phi_data.std().item(),
            'min': phi_data.min().item(),
            'max': phi_data.max().item(),
            'norm': phi_data.norm().item()
        }
    
    def get_kintsugi_state(self) -> Dict[str, Any]:
        """Get complete Kintsugi state information."""
        return {
            'epoch': self.epoch_count,
            'phi_stats': self.get_phi_stats(),
            'history_length': len(self.phi_history),
            'last_loss': self._last_loss,
            'momentum_norm': self._phi_momentum.norm().item() if self._phi_momentum is not None else 0.0,
            'name': self.name
        }
    
    def reset_history(self):
        """Clear phi and loss history."""
        self.phi_history.clear()
        self.loss_history.clear()
        self.epoch_count = 0
    
    def set_phi_modulation(self, enable: bool):
        """Enable or disable phi modulation."""
        self.use_phi_modulation = enable
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'name={self.name}, phi_shape={self.phi_shape}, '
                f'use_modulation={self.use_phi_modulation}, '
                f'kintsugi_lr={self.kintsugi_lr}')


# Example usage and helper functions
class SigilNet(nn.Module):
    """Example network using SigilLayer wrappers with proper architecture."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = SigilLayer(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            name="conv1",
            use_phi_modulation=True
        )
        
        self.conv2 = SigilLayer(
            nn.Conv2d(32, 64, 3, padding=1),
            name="conv2", 
            use_phi_modulation=True
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size (assuming 28x28 input like MNIST)
        # After 2 max pools: 28 -> 14 -> 7
        flattened_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = SigilLayer(
            nn.Linear(flattened_size, 128),
            name="fc1",
            use_phi_modulation=True
        )
        
        self.fc2 = SigilLayer(
            nn.Linear(128, num_classes),
            name="output",
            use_phi_modulation=False  # No modulation on output layer
        )
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: [batch, channels, height, width]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)  # No activation on output (handled by loss function)
        return x
    
    def update_all_sigils(self, loss):
        """Update all SigilLayers with current loss."""
        for module in self.modules():
            if isinstance(module, SigilLayer):
                module.update_phi(loss)
    
    def record_epoch(self, loss):
        """Record epoch for all SigilLayers."""
        for module in self.modules():
            if isinstance(module, SigilLayer):
                module.record_epoch(loss)
    
    def get_sigil_summary(self) -> Dict[str, Dict]:
        """Get summary of all SigilLayer states."""
        summary = {}
        for name, module in self.named_modules():
            if isinstance(module, SigilLayer):
                summary[name] = module.get_kintsugi_state()
        return summary


class SigilLinearNet(nn.Module):
    """Alternative example with purely linear layers for tabular data."""
    
    def __init__(self, input_dim: int = 784, hidden_dims: list = [256, 128], output_dim: int = 10):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            is_output = (i == len(dims) - 2)
            layer = SigilLayer(
                nn.Linear(dims[i], dims[i+1]),
                name=f"layer_{i}",
                use_phi_modulation=not is_output  # No modulation on output
            )
            self.layers.append(layer)
    
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # No activation on final layer
        x = self.layers[-1](x)
        return x
    
    def update_all_sigils(self, loss):
        """Update all SigilLayers with current loss."""
        for layer in self.layers:
            if isinstance(layer, SigilLayer):
                layer.update_phi(loss)
    
    def record_epoch(self, loss):
        """Record epoch for all SigilLayers.""" 
        for layer in self.layers:
            if isinstance(layer, SigilLayer):
                layer.record_epoch(loss)
