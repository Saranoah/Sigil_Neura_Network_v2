import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from collections import deque

class SigilLayer(nn.Module):
    """
    Practical wrapper for PyTorch layers with phi modulation and gradient-based updates.
    
    Args:
        base_layer: The PyTorch layer to wrap
        name: Optional name for the layer
        use_phi_modulation: Whether to apply phi modulation in forward pass
        kintsugi_lr: Learning rate for phi updates (default: 1e-4)
        history_limit: Maximum number of phi states to store (default: 100)
        phi_init_std: Standard deviation for phi initialization (default: 0.01)
        phi_shape: Optional explicit shape for phi (avoid auto-detection issues)
    """
    
    def __init__(
        self, 
        base_layer: nn.Module, 
        name: Optional[str] = None,
        use_phi_modulation: bool = True,
        kintsugi_lr: float = 1e-4,
        history_limit: int = 100,
        phi_init_std: float = 0.01,
        phi_shape: Optional[torch.Size] = None
    ):
        super().__init__()
        self.base_layer = base_layer
        self.name = name or f"Sigil_{base_layer.__class__.__name__}"
        self.use_phi_modulation = use_phi_modulation
        self.kintsugi_lr = kintsugi_lr
        self.history_limit = history_limit
        
        # Determine phi dimensions
        if phi_shape is not None:
            self.phi_shape = phi_shape
        else:
            self.phi_shape = self._get_phi_shape(base_layer)
        
        # Initialize phi parameter
        self.phi = nn.Parameter(torch.randn(self.phi_shape) * phi_init_std + 1.0)
        
        # Initialize momentum on the same device as phi
        self.register_buffer('_phi_momentum', torch.zeros_like(self.phi))
        
        # History tracking with efficient deque
        self.phi_history = deque(maxlen=history_limit)
        self.epoch_count = 0
        
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
            # Try to infer from a dummy forward pass
            try:
                with torch.no_grad():
                    dummy_input = self._create_dummy_input(layer)
                    if dummy_input is not None:
                        dummy_output = layer(dummy_input)
                        if len(dummy_output.shape) >= 2:
                            return torch.Size([dummy_output.shape[1]])
            except Exception as e:
                pass
            
            # If we can't determine shape, raise an error
            raise ValueError(
                f"Could not infer phi shape for layer {layer}. "
                "Please provide phi_shape explicitly in constructor."
            )
    
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
        
        # Handle different tensor shapes
        if len(x.shape) == 2:  # Linear layer output [batch, features]
            return x * phi.unsqueeze(0)
        elif len(x.shape) == 3:  # Conv1d output [batch, channels, length]
            return x * phi.unsqueeze(0).unsqueeze(-1)
        elif len(x.shape) == 4:  # Conv2d output [batch, channels, height, width]
            return x * phi.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif len(x.shape) == 5:  # Conv3d output [batch, channels, depth, height, width]
            return x * phi.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            # Fallback: broadcast along the second dimension
            phi_expanded = phi.view(1, -1, *([1] * (len(x.shape) - 2)))
            return x * phi_expanded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional phi modulation."""
        base_output = self.base_layer(x)
        return self._apply_phi_modulation(base_output)
    
    def update_phi(self):
        """
        Simplified phi update using only gradient information.
        Call this after backward() but before optimizer.step().
        """
        if not self.training or self.phi.grad is None:
            return
        
        # Apply momentum
        self._phi_momentum = (0.9 * self._phi_momentum + 0.1 * self.phi.grad.data)
        
        # Update phi with momentum
        self.phi.data -= self.kintsugi_lr * self._phi_momentum
        
        # Apply reasonable constraints
        self.phi.data.clamp_(0.1, 10.0)
        
        # Zero the gradient to prevent interference with main optimizer
        self.phi.grad = None
    
    def record_epoch(self):
        """Record current phi state at end of epoch."""
        self.epoch_count += 1
        self.phi_history.append(self.phi.detach().cpu().clone())
    
    def get_phi_stats(self) -> Dict[str, float]:
        """Get statistics about phi values."""
        phi_data = self.phi.detach()
        return {
            'mean': phi_data.mean().item(),
            'std': phi_data.std().item(),
            'min': phi_data.min().item(),
            'max': phi_data.max().item(),
        }


# Example networks with dynamic shape calculation
class SigilNet(nn.Module):
    """Example CNN using SigilLayer wrappers."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = SigilLayer(nn.Conv2d(input_channels, 32, 3, padding=1), name="conv1")
        self.conv2 = SigilLayer(nn.Conv2d(32, 64, 3, padding=1), name="conv2")
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dynamically calculate flattened size
        self.flattened_size = self._calculate_flattened_size(input_channels)
        
        # Fully connected layers
        self.fc1 = SigilLayer(nn.Linear(self.flattened_size, 128), name="fc1")
        self.fc2 = SigilLayer(nn.Linear(128, num_classes), name="output", use_phi_modulation=False)
        
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_flattened_size(self, input_channels):
        """Calculate the flattened size dynamically."""
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 28, 28)
            dummy_output = self.pool(self.conv1(dummy_input))
            dummy_output = self.pool(self.conv2(dummy_output))
            return dummy_output.view(1, -1).size(1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def update_all_sigils(self):
        """Update all SigilLayers."""
        for module in self.modules():
            if isinstance(module, SigilLayer):
                module.update_phi()
    
    def record_all_epochs(self):
        """Record epoch for all SigilLayers."""
        for module in self.modules():
            if isinstance(module, SigilLayer):
                module.record_epoch()


# Simple training example
def train_simple_example():
    """Simple training example to test the code."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SigilNet().to(device)
    
    # Create sample data (MNIST-like)
    x = torch.randn(32, 1, 28, 28).to(device)
    y = torch.randint(0, 10, (32,)).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop
    model.train()
    for epoch in range(3):  # Just 3 epochs for testing
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Update main parameters
        optimizer.step()
        
        # Update sigil parameters
        model.update_all_sigils()
        
        # Record epoch
        model.record_all_epochs()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        test_output = model(x)
        print(f"Test output shape: {test_output.shape}")
        print("Model works correctly!")

if __name__ == "__main__":
    train_simple_example()
