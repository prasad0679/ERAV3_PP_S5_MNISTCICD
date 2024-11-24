import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTNet
from src.utils import evaluate_model
from src.train import train_model

def test_model_parameters():
    model = MNISTNet()
    param_count = model.count_parameters()
    print(f"\nTotal number of model parameters: {param_count:,}")
    assert param_count < 25000, "Model has too many parameters"

def test_input_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape is incorrect"

def test_model_accuracy():
    # Use the model trained in the previous step
    model = MNISTNet()
    device = torch.device('cpu')
    
    # Find the most recent model file
    model_files = [f for f in os.listdir('.') if f.endswith('github.pth')]
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        # Use weights_only=True to avoid the warning
        model.load_state_dict(torch.load(latest_model, weights_only=True))
    else:
        model = train_model(device, save_suffix='test')[0]
    
    accuracy = evaluate_model(model, device)
    print(f"\nModel accuracy after 1 epoch: {accuracy:.2f}%")
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below threshold"

def test_output_probabilities():
    """Test if model outputs valid probability distributions"""
    model = MNISTNet()
    test_input = torch.randn(5, 1, 28, 28)
    with torch.no_grad():
        output = torch.softmax(model(test_input), dim=1)
    
    # Check if probabilities sum to 1 and are between 0 and 1
    sums = output.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5), "Output probabilities don't sum to 1"
    assert (output >= 0).all() and (output <= 1).all(), "Output contains invalid probabilities"
    print("\nModel output probability test passed")

def test_batch_processing():
    """Test if model can handle different batch sizes"""
    model = MNISTNet()
    batch_sizes = [1, 32, 64, 128]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    
    print("\nBatch processing test passed for sizes:", batch_sizes)

def test_model_gradients():
    """Test if model gradients are properly computed"""
    model = MNISTNet()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    test_input = torch.randn(16, 1, 28, 28)
    test_target = torch.randint(0, 10, (16,))
    
    # Training step
    optimizer.zero_grad()
    output = model(test_input)
    loss = criterion(output, test_target)
    loss.backward()
    
    # Check if gradients exist and are not zero
    has_gradients = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_gradients = True
                break
    
    assert has_gradients, "Model has no valid gradients"
    print("\nGradient computation test passed")