"""
Chaos Dynamics Module
Implements logistic map and Lorenz attractor for chaos injection into neural networks.
"""

import torch
import torch.nn as nn
import numpy as np


class LogisticMap(nn.Module):
    """
    Logistic map: x_{n+1} = r * x_n * (1 - x_n)
    
    Chaos regimes:
    - r < 3.0: Ordered (converges to fixed points)
    - r ≈ 3.57: Edge of chaos (period-doubling cascade)
    - r > 3.57: Chaotic (sensitive to initial conditions)
    """
    
    def __init__(self, r=3.5, num_steps=1):
        """
        Args:
            r: Chaos parameter (controls order vs. chaos)
            num_steps: Number of iterations to apply
        """
        super().__init__()
        self.r = r
        self.num_steps = num_steps
    
    def forward(self, x):
        """
        Apply logistic map transformation.
        
        Args:
            x: Input tensor (values should be in [0, 1])
        
        Returns:
            Transformed tensor
        """
        # Clamp to [0, 1] to ensure logistic map stays bounded
        x = torch.clamp(x, 0.0, 1.0)
        
        for _ in range(self.num_steps):
            x = self.r * x * (1 - x)
        
        return x


class LorenzAttractor(nn.Module):
    """
    Lorenz attractor: Chaotic system with three coupled ODEs.
    Used for more complex chaos dynamics.
    """
    
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
        """
        Args:
            sigma, rho, beta: Lorenz parameters
            dt: Time step for integration
        """
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
    
    def forward(self, x):
        """
        Apply Lorenz dynamics (simplified version for neural networks).
        
        Args:
            x: Input tensor (3D or reshaped to 3D)
        
        Returns:
            Transformed tensor
        """
        # Reshape to have 3 channels (for x, y, z components)
        if x.dim() == 2:
            # Flatten to 3D: [batch, features] -> [batch, 3, features/3]
            features = x.shape[1]
            if features % 3 != 0:
                # Pad if not divisible by 3
                pad_size = 3 - (features % 3)
                x = torch.nn.functional.pad(x, (0, pad_size))
                features = x.shape[1]
            x = x.view(x.shape[0], 3, features // 3)
        
        # Extract components (simplified - using first 3 channels)
        x_comp = x[:, 0, :]
        y_comp = x[:, 1, :] if x.shape[1] > 1 else x[:, 0, :]
        z_comp = x[:, 2, :] if x.shape[1] > 2 else x[:, 0, :]
        
        # Lorenz equations (simplified for neural network integration)
        dx = self.sigma * (y_comp - x_comp) * self.dt
        dy = (x_comp * (self.rho - z_comp) - y_comp) * self.dt
        dz = (x_comp * y_comp - self.beta * z_comp) * self.dt
        
        # Update
        x_new = x_comp + dx
        y_new = y_comp + dy
        z_new = z_comp + dz
        
        # Concatenate back
        if x.shape[1] >= 3:
            result = torch.stack([x_new, y_new, z_new], dim=1)
            result = result.view(result.shape[0], -1)
        else:
            result = x_new
        
        return result


class ChaosLayer(nn.Module):
    """
    Layer that injects chaos into neural network activations.
    Can use either logistic map or Lorenz attractor.
    """
    
    def __init__(self, chaos_type='logistic', r=3.5, strength=0.1, **kwargs):
        """
        Args:
            chaos_type: 'logistic' or 'lorenz'
            r: Chaos parameter for logistic map
            strength: How much chaos to inject (0.0 = no chaos, 1.0 = full chaos)
            **kwargs: Additional parameters for chaos dynamics
        """
        super().__init__()
        self.chaos_type = chaos_type
        self.strength = strength
        
        if chaos_type == 'logistic':
            self.chaos = LogisticMap(r=r, **kwargs)
        elif chaos_type == 'lorenz':
            self.chaos = LorenzAttractor(**kwargs)
        else:
            raise ValueError(f"Unknown chaos type: {chaos_type}")
    
    def forward(self, x):
        """
        Inject chaos into activations.
        
        Args:
            x: Input tensor
        
        Returns:
            x + chaos(x) * strength
        """
        # Normalize x to [0, 1] for logistic map
        if self.chaos_type == 'logistic':
            x_norm = torch.sigmoid(x)  # Map to [0, 1]
            chaos_signal = self.chaos(x_norm)
            # Map back and blend
            chaos_signal = torch.logit(torch.clamp(chaos_signal, 1e-7, 1-1e-7))
            return x + self.strength * chaos_signal
        else:
            # Lorenz: apply directly
            chaos_signal = self.chaos(x)
            return x + self.strength * (chaos_signal - x)

