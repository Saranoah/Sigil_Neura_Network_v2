import torch.nn as nn
from sigil_layer import SigilLayer
from sigil_network_manager import SigilNetworkManager

# Build your network with SigilLayer wrappers
model = nn.Sequential(
    SigilLayer(nn.Linear(128, 64), name="Encoder"),
    SigilLayer(nn.ReLU(), name="Activation1"),
    SigilLayer(nn.Linear(64, 32), name="Midlayer"),
    SigilLayer(nn.ReLU(), name="Activation2"),
    SigilLayer(nn.Linear(32, 10), name="Classifier")
)
manager = SigilNetworkManager(model)
for epoch in range(50):
    # ... training logic ...
    for layer in manager.sigil_layers:
        layer.update_phi()  # update phi after each epoch
    manager.record_epoch()
    if epoch % 10 == 0:
        manager.generate_gallery(epoch)

# Optionally, print summary metrics for README or dashboard
print(manager.summary_metrics())
