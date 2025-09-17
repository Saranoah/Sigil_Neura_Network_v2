import torch
import torch.nn as nn

class SigilLayer(nn.Module):
    """
    Wraps any torch.nn layer for Kintsugi metrics and sigil tracking.
    """
    def __init__(self, base_layer, name=None):
        super().__init__()
        self.base_layer = base_layer
        out_features = getattr(base_layer, 'out_features', 64)  # fallback
        self.phi = nn.Parameter(torch.ones(out_features))
        self.phi_history = []
        self.name = name if name else base_layer.__class__.__name__

    def forward(self, x):
        return self.base_layer(x)

    def update_phi(self, loss_map=None):
        # Placeholder for Kintsugi logic: update phi based on loss_map or error
        # For demonstration: random drift
        self.phi.data += torch.randn_like(self.phi) * 1e-3

    def record_epoch(self):
        self.phi_history.append(self.phi.detach().cpu().clone())

    def get_phi(self):
        return self.phi
