# Importing required libraries of pytorch (for model building) and sklearn (for evaluation metrics and split)
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import mean_absolute_percentage_error
from data_preprocessing import CustomDataset, transform

data_df = pd.read_csv("data/train.csv")

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)

# Create the dataset
train_dataset = CustomDataset(train_df, "data", transform=transform)

# Define the model architecture
class AgePredictionModel(nn.Module):
    def __init__(self):
        super(AgePredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AgePredictionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 20


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
total_absolute_percentage_error = 0
total_samples = 0

with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        outputs = model(inputs)

        absolute_percentage_error = np.abs((outputs.cpu().numpy() - labels.numpy()) / labels.numpy())
        total_absolute_percentage_error += np.sum(absolute_percentage_error)
        total_samples += len(labels)

# Calculate Mean Absolute Percentage Error (MAPE)
mape_train = (total_absolute_percentage_error / total_samples) * 100
print("Mean Absolute Percent Error (MAPE) on the training set:", mape_train)


# Save the trained model
torch.save(model.state_dict(), 'models/age_prediction_model.pth')



# Load the trained model
model = AgePredictionModel()
model.load_state_dict(torch.load('models/age_prediction_model.pth'))

# Create the test dataset
test_dataset = CustomDataset(val_df, "data", transform=transform)

# Create the test data loader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Evaluate the model on the validation set
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)

        # Convert output to numpy
        outputs = outputs.cpu().numpy()
        labels = labels.numpy()

        # Iterate over each prediction and label
        for output, label in zip(outputs, labels):
            # Print each prediction and label individually
            print("Predicted age: {:.2f}, Actual age: {:.2f}".format(output[0], label))
