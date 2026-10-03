"""
Edge of Chaos Neural Network
Neural network with tunable chaos injection layers.
"""

import torch
import torch.nn as nn
from chaos_dynamics import ChaosLayer


class EdgeOfChaosNetwork(nn.Module):
    """
    Neural network with chaos injection layers.
    The chaos parameter (r) controls the level of chaos in the network.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, 
                 chaos_type='logistic', r=3.5, chaos_strength=0.1, 
                 chaos_layers=None):
        """
        Args:
            input_size: Input dimension (e.g., 784 for MNIST)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            chaos_type: 'logistic' or 'lorenz'
            r: Chaos parameter (key tuning parameter)
            chaos_strength: How much chaos to inject (0.0-1.0)
            chaos_layers: Which layers to inject chaos into (None = all hidden layers)
        """
        super().__init__()
        
        self.chaos_type = chaos_type
        self.r = r
        self.chaos_strength = chaos_strength
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Inject chaos before activation
            if chaos_layers is None or i in chaos_layers:
                layers.append(ChaosLayer(
                    chaos_type=chaos_type,
                    r=r,
                    strength=chaos_strength
                ))
            
            # Activation
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_chaos_regime(self):
        """
        Determine the chaos regime based on r parameter.
        """
        if self.chaos_type == 'logistic':
            if self.r < 3.0:
                return "ordered"
            elif self.r < 3.57:
                return "transition"
            elif self.r < 3.9:
                return "edge_of_chaos"
            else:
                return "chaotic"
        else:
            return "lorenz_chaos"


def create_chaos_network(r, chaos_strength=0.1, **kwargs):
    """
    Factory function to create network with specific chaos parameter.
    
    Args:
        r: Chaos parameter
        chaos_strength: Strength of chaos injection
        **kwargs: Additional network parameters
    
    Returns:
        EdgeOfChaosNetwork instance
    """
    return EdgeOfChaosNetwork(r=r, chaos_strength=chaos_strength, **kwargs)

