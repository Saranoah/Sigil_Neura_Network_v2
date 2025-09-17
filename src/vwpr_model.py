import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque

class CorrectedKintsugiVWPR(nn.Module):
    """
    Corrected implementation of Value-Weighted Pathway Reinforcement
    with proper gradient flow and optimization mechanics.
    
    Implements the Kintsugi philosophy: honoring error as insight rather than minimizing it.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        phi_lr: float = 1e-3,
        phi_l1_weight: float = 1e-4,
        phi_normalization: bool = False,
        phi_clip_range: Tuple[float, float] = (1e-6, 10.0),
        history_limit: int = 100,
        softplus_beta: float = 1.0,
        layer_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.phi_lr = phi_lr
        self.phi_l1_weight = phi_l1_weight
        self.phi_normalization = phi_normalization
        self.phi_clip_range = phi_clip_range
        self.history_limit = history_limit
        self.softplus_beta = softplus_beta
        
        # Initialize pathway weights (phi) for each parameter - WITH GRADIENTS ENABLED
        self.pathway_weights = nn.ParameterDict()
        self._initialize_pathway_weights(layer_names)
        
        # Training history
        self.phi_history = defaultdict(lambda: deque(maxlen=history_limit))
        self.loss_history = deque(maxlen=history_limit)
        self.error_stats = {}
        self.step_count = 0
        
        # Current state
        self.last_weighted_loss = None
        self.total_pathway_mass = 0.0
        
    def _initialize_pathway_weights(self, layer_names: Optional[List[str]] = None):
        """Initialize phi weights for each trainable parameter WITH GRADIENTS ENABLED."""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and (layer_names is None or any(layer_name in name for layer_name in layer_names)):
                phi_name = f"phi_{name.replace('.', '_')}"
                # Start with small positive values - WITH GRADIENTS ENABLED
                self.pathway_weights[phi_name] = nn.Parameter(
                    torch.full_like(param.data, 0.1), requires_grad=True
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model."""
        return self.base_model(x)
    
    def _compute_local_errors(self, base_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute local error signals for each parameter.
        These are used to identify high-information pathways.
        """
        local_errors = {}
        
        # Clear previous gradients
        self.base_model.zero_grad()
        
        # Compute gradients
        base_loss.backward(retain_graph=True)
        
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Use gradient magnitude as error signal
                error_signal = param.grad.abs()
                local_errors[name] = error_signal.detach().clone()
        
        return local_errors
    
    def _apply_pathway_weighting(self, phi: torch.Tensor) -> torch.Tensor:
        """Apply softplus to ensure positive pathway weights with numerical stability."""
        return F.softplus(phi, beta=self.softplus_beta) + 1e-8
    
    def _compute_weighted_loss(
        self, 
        base_loss: torch.Tensor,
        local_errors: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute value-weighted loss by amplifying high-error pathways.
        """
        weighted_loss = base_loss.clone()
        phi_values = {}
        
        total_weighting = 0.0
        
        for name, error_signal in local_errors.items():
            phi_name = f"phi_{name.replace('.', '_')}"
            
            if phi_name in self.pathway_weights:
                # Get positive pathway weights
                raw_phi = self.pathway_weights[phi_name]
                phi_weighted = self._apply_pathway_weighting(raw_phi)
                
                # Clip to prevent runaway amplification
                phi_weighted = torch.clamp(phi_weighted, *self.phi_clip_range)
                
                # Ensure error_signal is a tensor with proper shape for broadcasting
                if error_signal.dim() == 0:  # If it's a scalar
                    error_signal = error_signal.view(1)  # Make it a 1D tensor
                
                # Weight the pathways by their error signals
                pathway_contribution = (phi_weighted * error_signal).sum()
                weighted_loss = weighted_loss + pathway_contribution
                
                phi_values[name] = phi_weighted.detach().clone()
                total_weighting += pathway_contribution.item()
        
        # Add L1 regularization to prevent overgrowth
        l1_penalty = sum(phi.abs().sum() for phi in self.pathway_weights.values())
        weighted_loss = weighted_loss + self.phi_l1_weight * l1_penalty
        
        self.total_pathway_mass = total_weighting
        return weighted_loss, phi_values
    
    def _update_pathway_weights(
        self, 
        local_errors: Dict[str, torch.Tensor],
        optimizer_phi: torch.optim.Optimizer
    ):
        """Update pathway weights to amplify high-error pathways using proper optimization."""
        
        phi_loss = 0.0
        
        for name, error_signal in local_errors.items():
            phi_name = f"phi_{name.replace('.', '_')}"
            
            if phi_name in self.pathway_weights:
                phi_weighted = self._apply_pathway_weighting(self.pathway_weights[phi_name])
                
                # Maximize pathway weights on high-error areas (gradient ascent)
                # We use negative to convert maximization to minimization
                phi_objective = -(phi_weighted * error_signal).sum()
                phi_loss = phi_loss + phi_objective
        
        # Add regularization
        l1_penalty = sum(phi.abs().sum() for phi in self.pathway_weights.values())
        phi_loss = phi_loss + self.phi_l1_weight * l1_penalty
        
        # Compute gradients and update using optimizer
        optimizer_phi.zero_grad()
        phi_loss.backward()
        optimizer_phi.step()
        
        # Optional normalization
        if self.phi_normalization:
            self._normalize_pathway_weights()
    
    def _normalize_pathway_weights(self):
        """Keep pathway weights balanced across layers."""
        with torch.no_grad():
            total_mass = sum(phi.sum().item() for phi in self.pathway_weights.values())
            if total_mass > 0:
                target_mass = len(self.pathway_weights) * 1.0
                scale_factor = target_mass / total_mass
                
                for phi in self.pathway_weights.values():
                    phi.data *= scale_factor
    
    def _update_error_stats(self, phi_values: Dict[str, torch.Tensor]):
        """Track pathway weight statistics for analysis."""
        for name, phi_tensor in phi_values.items():
            variance = phi_tensor.var().item()
            self.error_stats[name] = {
                'mean': phi_tensor.mean().item(),
                'std': phi_tensor.std().item(),
                'stability': 1.0 / (1.0 + variance)
            }
    
    def training_step(
        self,
        base_loss: torch.Tensor,
        optimizer_theta: torch.optim.Optimizer,
        optimizer_phi: torch.optim.Optimizer
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Execute dual-stream training step with proper gradient management:
        1. Update model parameters (theta) with weighted loss
        2. Update pathway weights (phi) to amplify high-error pathways
        """
        self.step_count += 1
        
        # Clear gradients
        optimizer_theta.zero_grad()
        optimizer_phi.zero_grad()
        
        # Compute local error signals
        local_errors = self._compute_local_errors(base_loss)
        
        # Compute value-weighted loss
        weighted_loss, phi_values = self._compute_weighted_loss(base_loss, local_errors)
        
        # Update model parameters with weighted loss
        weighted_loss.backward(retain_graph=True)
        optimizer_theta.step()
        
        # Update pathway weights
        self._update_pathway_weights(local_errors, optimizer_phi)
        
        # Update statistics
        self._update_error_stats(phi_values)
        self.last_weighted_loss = weighted_loss.item()
        
        return local_errors, phi_values
    
    def record_step(self):
        """Record current state for analysis."""
        # Store pathway weight evolution
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                phi_name = f"phi_{name.replace('.', '_')}"
                if phi_name in self.pathway_weights:
                    phi_weighted = self._apply_pathway_weighting(self.pathway_weights[phi_name])
                    self.phi_history[name].append(phi_weighted.mean().item())  # Store mean for efficiency
        
        # Record loss
        if self.last_weighted_loss is not None:
            self.loss_history.append(self.last_weighted_loss)
    
    def plot_pathway_distributions(self, save_path: Optional[str] = None):
        """Plot distributions of pathway weights for analysis."""
        n_params = len(self.pathway_weights)
        if n_params == 0:
            return
        
        cols = min(4, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if n_params == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if n_params > 1 else axes
        
        fig.suptitle(f'Pathway Weight Distributions - Step {self.step_count}', 
                    fontsize=14)
        
        for idx, (phi_name, phi_param) in enumerate(self.pathway_weights.items()):
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # Get pathway weights
            phi_values = self._apply_pathway_weighting(phi_param).detach().cpu().numpy().flatten()
            
            # Plot histogram
            n_bins = min(50, len(phi_values) // 10) if len(phi_values) > 100 else 20
            ax.hist(phi_values, bins=n_bins, alpha=0.7, color='blue', edgecolor='navy')
            
            # Labels and stats
            layer_name = phi_name.replace("phi_", "").replace("_", ".")
            ax.set_title(f'{layer_name}', fontsize=10)
            ax.set_xlabel('Pathway Weight')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = np.mean(phi_values)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'μ={mean_val:.3f}')
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_params, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        phi_stats = {}
        for name, param in self.pathway_weights.items():
            phi_values = self._apply_pathway_weighting(param).detach()
            phi_stats[name] = {
                'mean': phi_values.mean().item(),
                'std': phi_values.std().item(),
                'min': phi_values.min().item(),
                'max': phi_values.max().item(),
                'total_weight': phi_values.sum().item()
            }
        
        return {
            'step_count': self.step_count,
            'total_pathway_mass': self.total_pathway_mass,
            'last_weighted_loss': self.last_weighted_loss,
            'pathway_statistics': phi_stats,
            'error_stats': self.error_stats,
            'loss_history_length': len(self.loss_history),
            'recent_avg_loss': np.mean(list(self.loss_history)[-10:]) if self.loss_history else 0.0
        }
    
    def print_status_report(self):
        """Print training status."""
        stats = self.get_training_stats()
        
        print("=" * 50)
        print("KINTSUGI VWPR TRAINING STATUS")
        print("=" * 50)
        print(f"Training Steps: {stats['step_count']}")
        print(f"Total Pathway Mass: {stats['total_pathway_mass']:.4f}")
        print(f"Last Weighted Loss: {stats['last_weighted_loss']:.6f}")
        print(f"Recent Avg Loss: {stats['recent_avg_loss']:.6f}")
        print()
        
        print("Top 5 Layers by Pathway Weight:")
        phi_stats = stats['pathway_statistics']
        sorted_layers = sorted(phi_stats.items(), key=lambda x: x[1]['total_weight'], reverse=True)
        
        for name, layer_stats in sorted_layers[:5]:
            layer_name = name.replace('phi_', '').replace('_', '.')
            print(f"  {layer_name}:")
            print(f"    Total Weight: {layer_stats['total_weight']:.4f}")
            print(f"    Mean: {layer_stats['mean']:.4f}, Std: {layer_stats['std']:.4f}")
        
        print("=" * 50)
    
    def reset_history(self):
        """Clear all training history."""
        self.phi_history.clear()
        self.loss_history.clear()
        self.error_stats.clear()
        self.step_count = 0


# Ceremonial visualization extension
class KintsugiVisualizer:
    """Extension for creating Kintsugi-inspired visualizations of pathway weights."""
    
    def __init__(self, vwpr_model: CorrectedKintsugiVWPR):
        self.vwpr_model = vwpr_model
        self.archetype_definitions = {
            "The Oracle": {
                "conditions": lambda mean, std, total: std > 0.8 and 0.3 < mean < 0.7,
                "color": "#4444AA",
                "description": "Wisdom through pattern recognition and balance"
            },
            "The Sentinel": {
                "conditions": lambda mean, std, total: total > 1.5,
                "color": "#FFD700", 
                "description": "Stability and protection of core knowledge"
            },
            # Add more archetypes as needed
        }
    
    def generate_sigil(self, layer_name: str, step: int) -> plt.Figure:
        """Generate a Kintsugi-inspired sigil for a layer."""
        stats = self.vwpr_model.get_training_stats()
        
        if f"phi_{layer_name.replace('.', '_')}" not in stats['pathway_statistics']:
            return None
            
        layer_stats = stats['pathway_statistics'][f"phi_{layer_name.replace('.', '_')}"]
        mean, std, total = layer_stats['mean'], layer_stats['std'], layer_stats['total_weight']
        
        # Determine archetype
        archetype = "The Wanderer"  # Default
        color = "#00CCFF"
        
        for name, definition in self.archetype_definitions.items():
            if definition["conditions"](mean, std, total):
                archetype = name
                color = definition["color"]
                break
        
        # Create sigil
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Create pattern based on statistics
        theta = np.linspace(0, 2*np.pi, 100)
        radii = 0.5 + 0.3 * np.sin(5*theta) * std  # Pattern based on standard deviation
        
        x = radii * np.cos(theta)
        y = radii * np.sin(theta)
        
        # Plot with determined color
        ax.plot(x, y, color=color, linewidth=2, alpha=0.8)
        ax.fill(x, y, color=color, alpha=0.3)
        
        # Add text
        ax.text(0, 0, f"{layer_name}\n{archetype}\nStep: {step}", 
                ha='center', va='center', color='white', fontsize=10)
        
        return fig
