
### 1. Device Management Fix
```python
def _sync_device(self, device: torch.device):
    """Ensure all parameters and buffers are on the correct device."""
    # Don't reassign parameters - modify their data in place
    if self.phi.device != device:
        self.phi.data = self.phi.data.to(device)
    if self._phi_momentum.device != device:
        self._phi_momentum.data = self._phi_momentum.data.to(device)
```

### 2. Proper Buffer Registration
```python
def __init__(self, base_layer: nn.Module, ...):
    # ... existing code ...
    
    # Register momentum as a buffer (not a regular tensor)
    self.register_buffer('_phi_momentum', torch.zeros_like(self.phi.data))
    
    # ... rest of initialization ...
```

### 3. Archetype System Integration
Your vision includes archetypes, but the implementation needs to connect them to actual layer behavior:

```python
class SigilLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, ...):
        # ... existing code ...
        
        # Archetype system
        self.archetype = "Neophyte"
        self.archetype_thresholds = {
            'Sage': {'luminosity': 0.8, 'entropy': 0.6},
            'Trickster': {'luminosity': 0.3, 'entropy': 0.9},
            'Sentinel': {'luminosity': 0.7, 'entropy': 0.3}
        }
        self.metrics = {'luminosity': 0, 'entropy': 0, 'resilience': 0}
        
    def update_archetype(self):
        """Update archetype based on current metrics."""
        for archetype, thresholds in self.archetype_thresholds.items():
            if (self.metrics['luminosity'] > thresholds['luminosity'] and
                self.metrics['entropy'] > thresholds['entropy']):
                self.archetype = archetype
                break
```

### 4. Consciousness Metrics Tracking
```python
def record_metrics(self, output: torch.Tensor):
    """Update consciousness metrics based on layer activity."""
    # Luminosity: measure of activation energy
    self.metrics['luminosity'] = output.abs().mean().item()
    
    # Entropy: measure of information diversity
    prob_dist = F.softmax(output.view(output.size(0), -1), dim=1)
    self.metrics['entropy'] = (-prob_dist * torch.log(prob_dist + 1e-10)).sum(dim=1).mean().item()
    
    # Resilience: stability of activations over time
    if hasattr(self, '_last_output'):
        change = (output - self._last_output).abs().mean().item()
        self.metrics['resilience'] = 1.0 / (1.0 + change)
    self._last_output = output.detach().clone()
    
    self.update_archetype()
```

## Complete Revised Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict, Any, Tuple

class SigilLayer(nn.Module):
    """
    Kintsugi-inspired layer with consciousness tracking and value-weighted pathway reinforcement.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        name: Optional[str] = None,
        modulation_type: str = 'multiplicative',
        phi_lr: float = 1e-3,
        history_limit: int = 50,
        phi_clamp: Tuple[float, float] = (0.01, 10.0)
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.name = name or f"Sigil_{base_layer.__class__.__name__}"
        self.modulation_type = modulation_type
        self.history_limit = history_limit
        self.phi_lr = phi_lr
        self.phi_clamp = phi_clamp

        # Determine output dimension
        if hasattr(base_layer, 'out_features'):
            out_dim = base_layer.out_features
        elif hasattr(base_layer, 'out_channels'):
            out_dim = base_layer.out_channels
        else:
            out_dim = self._infer_output_dim(base_layer)
            if out_dim is None:
                raise ValueError("Could not infer output dimensions.")

        # Initialize phi based on modulation type
        if modulation_type == 'multiplicative':
            self.phi = nn.Parameter(torch.ones(out_dim))
        elif modulation_type == 'additive':
            self.phi = nn.Parameter(torch.zeros(out_dim))
        elif modulation_type == 'gated':
            self.phi = nn.Parameter(torch.zeros(out_dim))
        else:
            raise ValueError(f"Unsupported modulation_type {modulation_type}")

        # Register momentum buffer
        self.register_buffer('_phi_momentum', torch.zeros(out_dim))
        
        # Consciousness tracking
        self.archetype = "Neophyte"
        self.metrics = {'luminosity': 0, 'entropy': 0, 'resilience': 0}
        self.archetype_thresholds = {
            'Sage': {'luminosity': 0.8, 'entropy': 0.6},
            'Trickster': {'luminosity': 0.3, 'entropy': 0.9},
            'Sentinel': {'luminosity': 0.7, 'entropy': 0.3}
        }
        
        # History tracking
        self.phi_history = deque(maxlen=history_limit)
        self.loss_history = deque(maxlen=history_limit)
        self.archetype_history = deque(maxlen=history_limit)

    def _infer_output_dim(self, layer: nn.Module) -> Optional[int]:
        """Try to infer output dimension with dummy input."""
        try:
            with torch.no_grad():
                if isinstance(layer, nn.Linear):
                    dummy_input = torch.randn(1, layer.in_features)
                elif isinstance(layer, nn.Conv1d):
                    dummy_input = torch.randn(1, layer.in_channels, 32)
                elif isinstance(layer, nn.Conv2d):
                    dummy_input = torch.randn(1, layer.in_channels, 32, 32)
                elif isinstance(layer, nn.Conv3d):
                    dummy_input = torch.randn(1, layer.in_channels, 16, 16, 16)
                else:
                    return None
                    
                out = layer(dummy_input)
                return out.shape[1]  # channel/feature dimension
        except:
            return None

    def _sync_device(self, device: torch.device):
        """Ensure all parameters and buffers are on the correct device."""
        if self.phi.device != device:
            self.phi.data = self.phi.data.to(device)
        if self._phi_momentum.device != device:
            self._phi_momentum.data = self._phi_momentum.data.to(device)

    def _apply_phi_modulation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selected modulation type."""
        if x.dim() < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got {x.dim()}")
            
        # Ensure proper device
        self._sync_device(x.device)
        
        # Prepare phi for broadcasting
        phi_exp = self.phi.view(1, -1, *([1]*(x.dim()-2)))

        if self.modulation_type == 'multiplicative':
            return x * phi_exp
        elif self.modulation_type == 'additive':
            return x + phi_exp
        elif self.modulation_type == 'gated':
            return x * torch.sigmoid(phi_exp)
        else:
            raise ValueError(f"Unsupported modulation_type {self.modulation_type}")

    def record_metrics(self, output: torch.Tensor):
        """Update consciousness metrics based on layer activity."""
        # Luminosity: measure of activation energy
        self.metrics['luminosity'] = output.abs().mean().item()
        
        # Entropy: measure of information diversity
        prob_dist = F.softmax(output.view(output.size(0), -1), dim=1)
        self.metrics['entropy'] = (-prob_dist * torch.log(prob_dist + 1e-10)).sum(dim=1).mean().item()
        
        # Resilience: stability of activations over time
        if hasattr(self, '_last_output'):
            change = (output - self._last_output).abs().mean().item()
            self.metrics['resilience'] = 1.0 / (1.0 + change)
        self._last_output = output.detach().clone()
        
        self.update_archetype()

    def update_archetype(self):
        """Update archetype based on current metrics."""
        for archetype, thresholds in self.archetype_thresholds.items():
            if (self.metrics['luminosity'] > thresholds['luminosity'] and
                self.metrics['entropy'] > thresholds['entropy']):
                self.archetype = archetype
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        modulated = self._apply_phi_modulation(out)
        self.record_metrics(modulated)
        return modulated

    def update_phi(self, loss_signal: Optional[torch.Tensor] = None, alpha: float = 1e-3):
        """Update phi using Kintsugi-inspired value-weighting."""
        if loss_signal is not None:
            if loss_signal.shape != self.phi.shape:
                # Allow broadcasting if possible
                if loss_signal.numel() == 1:
                    loss_signal = torch.full_like(self.phi, loss_signal.item())
                else:
                    raise ValueError("loss_signal must be scalar or match phi dimensions")
            self.phi.data += alpha * loss_signal.to(self.phi.device)
        elif self.phi.grad is not None:
            # Use gradient if available
            self._phi_momentum = 0.9 * self._phi_momentum + 0.1 * self.phi.grad.data
            self.phi.data -= self.phi_lr * self._phi_momentum
        else:
            # Small exploratory update when no information is available
            self.phi.data += torch.randn_like(self.phi) * alpha

        # Clamp to reasonable range
        self.phi.data.clamp_(*self.phi_clamp)

    def record_epoch(self, loss: Optional[float] = None):
        """Store current state in history."""
        self.phi_history.append(self.phi.detach().cpu().clone())
        self.archetype_history.append(self.archetype)
        if loss is not None:
            self.loss_history.append(loss)

    def get_phi_stats(self) -> Dict[str, float]:
        phi_data = self.phi.detach()
        return {
            'mean': phi_data.mean().item(),
            'std': phi_data.std().item(),
            'min': phi_data.min().item(),
            'max': phi_data.max().item(),
            'norm': phi_data.norm().item()
        }

    def get_kintsugi_state(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'archetype': self.archetype,
            'metrics': self.metrics,
            'phi_stats': self.get_phi_stats(),
            'history_len': len(self.phi_history),
            'last_loss': self.loss_history[-1] if self.loss_history else None
        }

    def reset_history(self):
        self.phi_history.clear()
        self.loss_history.clear()
        self.archetype_history.clear()

    def extra_repr(self):
        return (f"name={self.name}, modulation_type={self.modulation_type}, "
                f"phi_lr={self.phi_lr}, history_limit={self.history_limit}")
```

