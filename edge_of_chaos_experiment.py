"""
Edge of Chaos Neural Networks Experiment
Tests whether neural networks perform best at the edge of chaos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

from edge_of_chaos_network import create_chaos_network


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ExperimentRunner:
    """Runs the edge of chaos experiment across multiple chaos levels."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = []
        
        # Setup data
        self.train_loader, self.test_loader = self._setup_data()
    
    def _setup_data(self, batch_size=64):
        """Setup MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, model, epochs=5, lr=0.001):
        """Train a single model."""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, 
                                                          desc=f"Epoch {epoch+1}/{epochs}", 
                                                          leave=False)):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
        
        return train_losses
    
    def evaluate_model(self, model):
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def run_experiment(self, r_values, epochs=5, chaos_strength=0.1):
        """
        Run experiment across multiple chaos levels.
        
        Args:
            r_values: List of chaos parameters (r) to test
            epochs: Number of training epochs
            chaos_strength: Strength of chaos injection
        """
        print(f"Running Edge of Chaos Experiment")
        print(f"Testing {len(r_values)} chaos levels: {r_values}")
        print(f"Device: {self.device}\n")
        
        for r in r_values:
            print(f"\n{'='*60}")
            print(f"Training model with r = {r:.2f}")
            print(f"{'='*60}")
            
            # Create model
            model = create_chaos_network(
                r=r,
                input_size=784,
                hidden_sizes=[128, 64],
                num_classes=10,
                chaos_strength=chaos_strength
            )
            
            # Get chaos regime
            regime = model.get_chaos_regime()
            print(f"Chaos regime: {regime}")
            
            # Train
            train_losses = self.train_model(model, epochs=epochs)
            
            # Evaluate
            accuracy = self.evaluate_model(model)
            
            print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
            
            # Store results
            result = {
                'r': r,
                'regime': regime,
                'accuracy': accuracy,
                'train_losses': train_losses,
                'final_loss': train_losses[-1] if train_losses else None
            }
            self.results.append(result)
        
        return self.results
    
    def plot_results(self, save_path='edge_of_chaos_results.png'):
        """Plot performance vs. chaos parameter."""
        if not self.results:
            print("No results to plot!")
            return
        
        # Extract data
        r_values = [r['r'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        regimes = [r['regime'] for r in self.results]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color by regime
        color_map = {
            'ordered': 'blue',
            'transition': 'green',
            'edge_of_chaos': 'red',
            'chaotic': 'orange',
            'lorenz_chaos': 'purple'
        }
        
        colors = [color_map.get(regime, 'gray') for regime in regimes]
        
        # Plot
        ax.scatter(r_values, accuracies, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax.plot(r_values, accuracies, 'k--', alpha=0.3, linewidth=1)
        
        # Add labels
        for i, (r, acc, regime) in enumerate(zip(r_values, accuracies, regimes)):
            ax.annotate(f'{acc:.1f}%', (r, acc), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        # Formatting
        ax.set_xlabel('Chaos Parameter (r)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Edge of Chaos: Performance vs. Chaos Parameter', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[regime], label=regime.replace('_', ' ').title()) 
                          for regime in set(regimes) if regime in color_map]
        ax.legend(handles=legend_elements, loc='best')
        
        # Highlight edge of chaos region (r ≈ 3.57)
        ax.axvline(x=3.57, color='red', linestyle='--', alpha=0.5, 
                  label='Theoretical Edge of Chaos')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {save_path}")
        plt.show()
    
    def save_results(self, save_path='edge_of_chaos_results.json'):
        """Save results to JSON file."""
        # Convert numpy/torch types to native Python types
        results_serializable = []
        for r in self.results:
            result = {
                'r': float(r['r']),
                'regime': r['regime'],
                'accuracy': float(r['accuracy']),
                'final_loss': float(r['final_loss']) if r['final_loss'] is not None else None,
                'train_losses': [float(loss) for loss in r['train_losses']]
            }
            results_serializable.append(result)
        
        with open(save_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results_serializable
            }, f, indent=2)
        
        print(f"Results saved to {save_path}")


def main():
    """Main experiment execution."""
    print("="*60)
    print("EDGE OF CHAOS NEURAL NETWORKS EXPERIMENT")
    print("="*60)
    print("\nTesting hypothesis: Neural networks perform best at edge of chaos")
    print("(r ≈ 3.57 for logistic map)\n")
    
    # Define chaos levels to test
    # Ordered regime
    r_ordered = [2.0, 2.5]
    # Transition regime
    r_transition = [3.0, 3.2, 3.4]
    # Edge of chaos
    r_edge = [3.5, 3.57, 3.6, 3.7]
    # Chaotic regime
    r_chaotic = [3.8, 3.9, 4.0]
    
    # Combine all
    r_values = r_ordered + r_transition + r_edge + r_chaotic
    r_values = sorted(r_values)  # Sort for better visualization
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Run experiment
    results = runner.run_experiment(r_values, epochs=5, chaos_strength=0.1)
    
    # Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest Performance:")
    print(f"  r = {best_result['r']:.2f}")
    print(f"  Regime: {best_result['regime']}")
    print(f"  Accuracy: {best_result['accuracy']:.2f}%")
    
    # Find edge of chaos performance
    edge_results = [r for r in results if r['regime'] == 'edge_of_chaos']
    if edge_results:
        edge_avg = np.mean([r['accuracy'] for r in edge_results])
        print(f"\nAverage Edge of Chaos Performance: {edge_avg:.2f}%")
    
    # Plot and save
    runner.plot_results()
    runner.save_results()
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

