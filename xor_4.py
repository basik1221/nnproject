import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
        self.fc1 = nn.Linear(2, 2)
        self.activation = nn.Sigmoid() 
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a single-layer neural network
class SingleLayerModel(nn.Module):
    def __init__(self):
        super(SingleLayerModel, self).__init__()
        self.fc = nn.Linear(2, 1)  # Input layer to output layer
        self.activation = nn.Sigmoid()  

    def forward(self, x):
        x = self.activation(self.fc(x))
        return x

# Define a three-layer neural network 
class ThreeLayerModel(nn.Module):
    def __init__(self):
        super(ThreeLayerModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # Input layer to hidden layer 1
        self.activation1 = nn.Sigmoid()  # 
        self.fc2 = nn.Linear(8, 8)  # Hidden layer 1 to hidden layer 2
        self.activation2 = nn.Sigmoid()  
        self.fc3 = nn.Linear(8, 1)  # Hidden layer 2 to output layer

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a four-layer neural network
class FourLayerModel(nn.Module):
    def __init__(self):
        super(FourLayerModel, self).__init__()
        self.fc1 = nn.Linear(2, 9)  
        self.activation1 = nn.Sigmoid()
        self.fc2 = nn.Linear(9, 9) 
        self.activation2 = nn.Sigmoid()
        self.fc3 = nn.Linear(9, 9)  
        self.activation3 = nn.Sigmoid()
        self.fc4 = nn.Linear(9, 1)  

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.fc4(x)
        return x

# Define a five-layer neural network
class FiveLayerModel(nn.Module):
    def __init__(self):
        super(FiveLayerModel, self).__init__()
        self.fc1 = nn.Linear(2, 9)  
        self.activation1 = nn.Sigmoid()
        self.fc2 = nn.Linear(9, 9)  
        self.activation2 = nn.Sigmoid()
        self.fc3 = nn.Linear(9, 9)  
        self.activation3 = nn.Sigmoid()
        self.fc4 = nn.Linear(9, 9)  
        self.activation4 = nn.Sigmoid()
        self.fc5 = nn.Linear(9, 1)  

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.activation4(self.fc4(x))
        x = self.fc5(x)
        return x


# Define a six-layer neural network
class SixLayerModel(nn.Module):
    def __init__(self):
        super(SixLayerModel, self).__init__()
        self.fc1 = nn.Linear(2, 9)  
        self.activation1 = nn.Sigmoid()  
        self.fc2 = nn.Linear(9, 9)  
        self.activation2 = nn.Sigmoid()  
        self.fc3 = nn.Linear(9, 9) 
        self.activation3 = nn.Sigmoid() 
        self.fc4 = nn.Linear(9, 9)  
        self.activation4 = nn.Sigmoid()  
        self.fc5 = nn.Linear(9, 9)  
        self.activation5 = nn.Sigmoid()  
        self.fc6 = nn.Linear(9, 1) 
    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation3(self.fc3(x))
        x = self.activation4(self.fc4(x))
        x = self.activation5(self.fc5(x))
        x = self.fc6(x)
        return x



# Define the plot_decision_boundary function
def plot_decision_boundary(model, data_inputs, title):
    # Adjusted range for the plot
    x_min, x_max = -15.5, 15.5
    y_min, y_max = -15.5, 15.5

    # Prepare grid of points in input space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_input = np.c_[xx.ravel(), yy.ravel()]

    # Use the model to predict the class for each point in the grid
    with torch.no_grad():
        model.eval()
        predictions = model(torch.tensor(grid_input, dtype=torch.float32))
        predictions = (predictions > 0.5).float().numpy().reshape(xx.shape)

    plt.contourf(xx, yy, predictions, cmap=plt.cm.Blues, alpha=0.5, levels=[-0.5, 0.5, 1.5])
    plt.scatter(data_inputs[:, 0], data_inputs[:, 1], c=targets.numpy(), cmap=plt.cm.Blues, edgecolors='k')
    plt.title(title)
    plt.show()


# Instantiate the models, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {'XORModel': XORModel(), 'SingleLayerModel': SingleLayerModel(),
          'ThreeLayerModel': ThreeLayerModel(), 'FourLayerModel': FourLayerModel(),
          'FiveLayerModel': FiveLayerModel(), 'SixLayerModel': SixLayerModel()}
criterion = nn.MSELoss()
optimizers = {name: optim.SGD(model.parameters(), lr=0.1) for name, model in models.items()}

# Training loop for each model
epochs = 15000
for model_name, model in models.items():
    print(f"\nTraining {model_name}:")
    correct_count = 0
    incorrect_count = 0

    for epoch in range(epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizers[model_name].zero_grad()
        loss.backward()
        optimizers[model_name].step()

        # Count correct and incorrect predictions
        predictions = (outputs > 0.5).float()
        correct_count += torch.sum(predictions == targets)
        incorrect_count += torch.sum(predictions != targets)

        if epoch % 1000 == 0:
            # Calculate accuracy
            accuracy = correct_count.item() / (correct_count.item() + incorrect_count.item())
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy:.2%}')

    accuracy = correct_count.item() / (correct_count.item() + incorrect_count.item())

    print(f"\n{model_name} - Correct Choices: {correct_count.item()}, Incorrect Choices: {incorrect_count.item()}, Accuracy: {accuracy:.2%}")

    # Plot decision boundary for each model
    with torch.no_grad():
        test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        plot_decision_boundary(model, test_inputs.numpy(), f'Decision Boundary for {model_name} - Accuracy: {accuracy:.2%} ReLU')

# Test the trained models
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    for model_name, model in models.items():
        print(f"\nTrained {model_name} Predictions:")
        predictions = model(test_inputs)
        predictions = (predictions > 0.5).float()  

        for i in range(len(test_inputs)):
            print(f"Input: {test_inputs[i].tolist()}, Predicted Output: {predictions[i].item()}")
