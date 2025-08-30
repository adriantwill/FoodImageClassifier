import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import numpy as np

# Define a simple transformation pipeline (normalization and resizing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the Food-101 dataset folder (should contain 'train' and 'test' subdirectories)
data_dir = './food-101/'

# Load datasets using ImageFolder (will automatically split into train and test)
train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# Create data loaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Freeze the parameters of the pre-trained layers (optional for fine-tuning)
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to match the number of food categories (101 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 101)  # Food-101 has 101 classes

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Training the model
num_epochs = 5

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

# Call the training function
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')

# Evaluate the model after training
evaluate_model(model, test_loader)

# Saving the trained model (optional)
torch.save(model.state_dict(), 'food101_resnet50.pth')

# Sample image prediction
def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = train_dataset.classes[predicted.item()]
    return predicted_class

# Test prediction with a sample image
from PIL import Image
sample_image_path = './sample_image.jpg'  # Replace with an actual image path
predicted_food = predict_image(sample_image_path, model)
print(f'The food in the image is predicted to be: {predicted_food}')
