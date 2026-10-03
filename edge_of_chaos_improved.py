"""
Improved Edge of Chaos Experiment
Tests multiple improvements to find optimal chaos configuration.
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
from itertools import product
import argparse


DATASET_MAP = {
    'mnist': datasets.MNIST,
    'fashion-mnist': datasets.FashionMNIST,
}

DATASET_LABELS = {
    'mnist': 'MNIST',
    'fashion-mnist': 'Fashion-MNIST',
}

from edge_of_chaos_network import create_chaos_network


torch.manual_seed(42)
np.random.seed(42)


class ImprovedExperimentRunner:
    """Enhanced experiment runner with multiple optimization strategies."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 dataset='mnist', batch_size=64, data_root='./data'):
        self.device = device
        self.dataset = dataset.lower()
        if self.dataset not in DATASET_MAP:
            raise ValueError(f"Unsupported dataset: {dataset}")
        self.dataset_cls = DATASET_MAP[self.dataset]
        self.dataset_name = DATASET_LABELS[self.dataset]
        self.batch_size = batch_size
        self.data_root = data_root
        self.results = []
        self.train_loader, self.test_loader = self._setup_data()
    
    def _setup_data(self):
        """Setup dataset loaders based on configuration."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset_kwargs = dict(root=self.data_root, download=True, transform=transform)
        train_dataset = self.dataset_cls(train=True, **dataset_kwargs)
        test_dataset = self.dataset_cls(train=False, **dataset_kwargs)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, model, epochs=10, lr=0.001):
        """Train model with optional adaptive chaos."""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for data, target in tqdm(self.train_loader, 
                                    desc=f"Epoch {epoch+1}/{epochs}", 
                                    leave=False):
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
    
    def experiment_1_fine_tune_r(self, epochs=10):
        """
        Improvement 1: Fine-tune r parameter around optimal region (r ≈ 3.2).
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 1: Fine-tuning r parameter around optimal region")
        print("="*70)
        
        # More granular values around r = 3.2
        r_values = np.linspace(3.0, 3.5, 11)  # 11 values between 3.0 and 3.5
        
        results = []
        for r in r_values:
            print(f"\nTraining with r = {r:.3f}")
            
            model = create_chaos_network(
                r=r,
                input_size=784,
                hidden_sizes=[128, 64],
                num_classes=10,
                chaos_strength=0.1
            )
            
            regime = model.get_chaos_regime()
            train_losses = self.train_model(model, epochs=epochs)
            accuracy = self.evaluate_model(model)
            
            print(f"  Regime: {regime}, Accuracy: {accuracy:.2f}%")
            
            results.append({
                'experiment': 'fine_tune_r',
                'dataset': self.dataset_name,
                'r': float(r),
                'regime': regime,
                'accuracy': float(accuracy),
                'final_loss': float(train_losses[-1])
            })
        
        return results
    
    def experiment_2_chaos_strength(self, epochs=10):
        """
        Improvement 2: Test different chaos injection strengths.
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 2: Testing different chaos injection strengths")
        print("="*70)
        
        # Keep r at optimal (3.2), vary strength
        r_optimal = 3.2
        strength_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        
        results = []
        for strength in strength_values:
            print(f"\nTraining with r = {r_optimal:.2f}, strength = {strength:.2f}")
            
            model = create_chaos_network(
                r=r_optimal,
                input_size=784,
                hidden_sizes=[128, 64],
                num_classes=10,
                chaos_strength=strength
            )
            
            train_losses = self.train_model(model, epochs=epochs)
            accuracy = self.evaluate_model(model)
            
            print(f"  Accuracy: {accuracy:.2f}%")
            
            results.append({
                'experiment': 'chaos_strength',
                'dataset': self.dataset_name,
                'r': float(r_optimal),
                'strength': float(strength),
                'accuracy': float(accuracy),
                'final_loss': float(train_losses[-1])
            })
        
        return results
    
    def experiment_3_layer_specific_chaos(self, epochs=10):
        """
        Improvement 3: Test chaos injection in different layers.
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 3: Testing layer-specific chaos injection")
        print("="*70)
        
        r_optimal = 3.2
        strength = 0.1
        
        # Test different layer configurations
        layer_configs = [
            None,  # All layers
            [0],   # First layer only
            [1],   # Second layer only
            [0, 1] # Both layers (same as None for 2-layer network)
        ]
        
        config_names = ["All layers", "First layer only", "Second layer only", "Both layers"]
        
        results = []
        for config, name in zip(layer_configs, config_names):
            print(f"\nTraining with {name}")
            
            # Need to modify network creation for layer-specific chaos
            from edge_of_chaos_network import EdgeOfChaosNetwork
            
            model = EdgeOfChaosNetwork(
                r=r_optimal,
                input_size=784,
                hidden_sizes=[128, 64],
                num_classes=10,
                chaos_strength=strength,
                chaos_layers=config
            )
            
            train_losses = self.train_model(model, epochs=epochs)
            accuracy = self.evaluate_model(model)
            
            print(f"  Accuracy: {accuracy:.2f}%")
            
            results.append({
                'experiment': 'layer_specific',
                'dataset': self.dataset_name,
                'r': float(r_optimal),
                'strength': float(strength),
                'chaos_layers': str(config),
                'layer_config': name,
                'accuracy': float(accuracy),
                'final_loss': float(train_losses[-1])
            })
        
        return results
    
    def experiment_4_optimal_combination(self, epochs=10):
        """
        Improvement 4: Find optimal combination of r and strength.
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 4: Finding optimal r and strength combination")
        print("="*70)
        
        # Grid search over promising ranges
        r_values = np.linspace(3.1, 3.4, 7)  # Around optimal r
        strength_values = [0.05, 0.1, 0.15, 0.2]  # Promising strengths
        
        best_accuracy = 0
        best_config = None
        results = []
        
        print(f"\nTesting {len(r_values)} r values × {len(strength_values)} strengths = {len(r_values) * len(strength_values)} configurations")
        
        for r, strength in product(r_values, strength_values):
            print(f"\nTraining: r = {r:.2f}, strength = {strength:.2f}")
            
            model = create_chaos_network(
                r=r,
                input_size=784,
                hidden_sizes=[128, 64],
                num_classes=10,
                chaos_strength=strength
            )
            
            train_losses = self.train_model(model, epochs=epochs)
            accuracy = self.evaluate_model(model)
            
            print(f"  Accuracy: {accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {
                    'dataset': self.dataset_name,
                    'r': r,
                    'strength': strength,
                    'accuracy': accuracy
                }
            
            results.append({
                'experiment': 'optimal_combination',
                'dataset': self.dataset_name,
                'r': float(r),
                'strength': float(strength),
                'accuracy': float(accuracy),
                'final_loss': float(train_losses[-1])
            })
        
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION FOUND:")
        print(f"  r = {best_config['r']:.3f}")
        print(f"  strength = {best_config['strength']:.3f}")
        print(f"  accuracy = {best_config['accuracy']:.2f}%")
        print("="*70)
        
        return results, best_config
    
    def plot_improvements(self, all_results):
        """Plot all improvement experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Experiment 1: Fine-tune r
        exp1 = [r for r in all_results if r.get('experiment') == 'fine_tune_r']
        if exp1:
            ax = axes[0, 0]
            r_vals = [r['r'] for r in exp1]
            accs = [r['accuracy'] for r in exp1]
            ax.plot(r_vals, accs, 'o-', linewidth=2, markersize=8)
            ax.axvline(x=3.2, color='red', linestyle='--', alpha=0.5, label='Previous optimal')
            ax.set_xlabel('Chaos Parameter (r)', fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title('Fine-tuned r Parameter', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Experiment 2: Chaos strength
        exp2 = [r for r in all_results if r.get('experiment') == 'chaos_strength']
        if exp2:
            ax = axes[0, 1]
            strengths = [r['strength'] for r in exp2]
            accs = [r['accuracy'] for r in exp2]
            ax.plot(strengths, accs, 's-', linewidth=2, markersize=8, color='green')
            ax.set_xlabel('Chaos Strength', fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title('Chaos Injection Strength', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Experiment 3: Layer-specific
        exp3 = [r for r in all_results if r.get('experiment') == 'layer_specific']
        if exp3:
            ax = axes[1, 0]
            configs = [r['layer_config'] for r in exp3]
            accs = [r['accuracy'] for r in exp3]
            ax.bar(range(len(configs)), accs, color=['blue', 'orange', 'green', 'red'])
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title('Layer-Specific Chaos Injection', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Experiment 4: Optimal combination (heatmap)
        exp4 = [r for r in all_results if r.get('experiment') == 'optimal_combination']
        if exp4:
            ax = axes[1, 1]
            r_vals = sorted(set(r['r'] for r in exp4))
            s_vals = sorted(set(r['strength'] for r in exp4))
            acc_matrix = np.zeros((len(s_vals), len(r_vals)))
            
            for r in exp4:
                r_idx = r_vals.index(r['r'])
                s_idx = s_vals.index(r['strength'])
                acc_matrix[s_idx, r_idx] = r['accuracy']
            
            im = ax.imshow(acc_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xticks(range(len(r_vals)))
            ax.set_xticklabels([f"{r:.2f}" for r in r_vals], rotation=45)
            ax.set_yticks(range(len(s_vals)))
            ax.set_yticklabels([f"{s:.2f}" for s in s_vals])
            ax.set_xlabel('Chaos Parameter (r)', fontweight='bold')
            ax.set_ylabel('Chaos Strength', fontweight='bold')
            ax.set_title('Optimal r × Strength Combination', fontweight='bold')
            plt.colorbar(im, ax=ax, label='Accuracy (%)')
        
        plt.tight_layout()
        plot_path = f'edge_of_chaos_improvements_{self.dataset}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {plot_path}")
        plt.show()
    
    def save_results(self, all_results, best_config, save_path=None):
        """Save all results to JSON."""
        if save_path is None:
            save_path = f'edge_of_chaos_improved_results_{self.dataset}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'dataset': self.dataset_name,
                'best_configuration': best_config,
                'results': all_results
            }, f, indent=2)
        
        print(f"Results saved to {save_path}")


def main():
    """Run improved experiments."""
    parser = argparse.ArgumentParser(description="Edge of Chaos improvement experiments")
    parser.add_argument('--dataset', choices=tuple(DATASET_MAP.keys()), default='mnist',
                        help='Dataset to run experiments on')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per configuration')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training and evaluation')
    parser.add_argument('--data-root', default='./data',
                        help='Root directory for datasets')
    args = parser.parse_args()
    
    print("="*70)
    print("IMPROVED EDGE OF CHAOS EXPERIMENTS")
    print("="*70)
    print(f"\nTesting multiple improvements to find optimal configuration on {DATASET_LABELS[args.dataset]}\n")
    
    runner = ImprovedExperimentRunner(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      data_root=args.data_root)
    
    all_results = []
    
    # Run all improvement experiments
    results1 = runner.experiment_1_fine_tune_r(epochs=args.epochs)
    all_results.extend(results1)
    
    results2 = runner.experiment_2_chaos_strength(epochs=args.epochs)
    all_results.extend(results2)
    
    results3 = runner.experiment_3_layer_specific_chaos(epochs=args.epochs)
    all_results.extend(results3)
    
    results4, best_config = runner.experiment_4_optimal_combination(epochs=args.epochs)
    all_results.extend(results4)
    
    # Plot and save
    runner.plot_improvements(all_results)
    runner.save_results(all_results, best_config)
    
    print("\n" + "="*70)
    print("All improvement experiments complete!")
    print("="*70)


if __name__ == '__main__':
    main()

