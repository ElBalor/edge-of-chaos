"""
Test training efficiency: Compare baseline vs chaos-injected network.
Measures: training time, convergence rate, final accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from edge_of_chaos_network import create_chaos_network

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading MNIST dataset (full)…")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

def train_and_measure(model, epochs=10, target_accuracy=97.5):
    """Train model and measure efficiency metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    train_losses = []
    accuracies = []
    epoch_to_target = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        accuracies.append(accuracy)
        
        if epoch_to_target is None and accuracy >= target_accuracy:
            epoch_to_target = epoch + 1
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'epoch_to_target': epoch_to_target,
        'final_accuracy': accuracies[-1],
        'train_losses': train_losses,
        'accuracies': accuracies
    }

# Test baseline (no chaos)
print("="*70)
print("BASELINE NETWORK (No Chaos)")
print("="*70)
baseline_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
baseline_results = train_and_measure(baseline_model, epochs=10)

# Test optimal chaos network
print("\n" + "="*70)
print("OPTIMAL CHAOS NETWORK (r=3.40, strength=0.15)")
print("="*70)
chaos_model = create_chaos_network(
    r=3.40,
    input_size=784,
    hidden_sizes=[128, 64],
    num_classes=10,
    chaos_strength=0.15
)
chaos_results = train_and_measure(chaos_model, epochs=10)

# Compare results
print("\n" + "="*70)
print("EFFICIENCY COMPARISON")
print("="*70)
print(f"\nBaseline:")
print(f"  Total training time: {baseline_results['total_time']:.2f} seconds")
print(f"  Final accuracy: {baseline_results['final_accuracy']:.2f}%")
print(f"  Epochs to reach 97.5%: {baseline_results['epoch_to_target'] or 'Not reached'}")

print(f"\nChaos Network:")
print(f"  Total training time: {chaos_results['total_time']:.2f} seconds")
print(f"  Final accuracy: {chaos_results['final_accuracy']:.2f}%")
print(f"  Epochs to reach 97.5%: {chaos_results['epoch_to_target'] or 'Not reached'}")

print(f"\nComparison:")
time_overhead = ((chaos_results['total_time'] / baseline_results['total_time']) - 1) * 100
accuracy_gain = chaos_results['final_accuracy'] - baseline_results['final_accuracy']
print(f"  Time overhead: {time_overhead:+.1f}%")
print(f"  Accuracy gain: {accuracy_gain:+.2f}%")

if chaos_results['epoch_to_target'] and baseline_results['epoch_to_target']:
    convergence_speedup = ((baseline_results['epoch_to_target'] / chaos_results['epoch_to_target']) - 1) * 100
    print(f"  Convergence speedup: {convergence_speedup:+.1f}%")
    print(f"    (Baseline: {baseline_results['epoch_to_target']} epochs, Chaos: {chaos_results['epoch_to_target']} epochs)")

