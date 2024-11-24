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