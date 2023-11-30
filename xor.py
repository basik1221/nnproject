import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define the XOR dataset
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XORDataset(Dataset):
    def __init__(self):
        self.inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        self.targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'target': self.targets[idx]
        }

# Define the XOR neural network
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input layer to hidden layer
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(2, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = XORModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# Test the trained model
with torch.no_grad():
    dataset = XORDataset
    batch_size = 2
    shuffle = True
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=shuffle)
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predictions = model(test_inputs)
    predictions = (predictions > 0.5).float()  # Convert to binary predictions

    print("\nTrained Model Predictions:")
    for i in range(len(test_inputs)):
        print(f"Input: {test_inputs[i].tolist()}, Predicted Output: {predictions[i].item()}")
