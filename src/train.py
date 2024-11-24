import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import platform
from src.model import MNISTNet
from src.utils import evaluate_model
import os
from tqdm import tqdm
from torchsummary import summary

def train_model(device='cpu', save_suffix='local'):
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    model = MNISTNet().to(device)
    
    # Print model summary
    print("\n=== Model Architecture ===")
    summary(model, input_size=(1, 28, 28))
    
    # Print total parameters
    total_params = model.count_parameters()
    print(f"\nTotal number of model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # Training
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate accuracy
    accuracy = evaluate_model(model, device)
    
    avg_loss = total_loss / len(train_loader)
    print(f"\n=== Training Summary ===")
    print(f"Total Parameters: {total_params:,}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"Batch Size: 512")
    
    # Save model with timestamp and system info
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    system_info = platform.system().lower()
    model_path = f'model_{timestamp}_{system_info}_{save_suffix}.pth'
    torch.save(model.state_dict(), model_path)
    return model, model_path

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(device) 