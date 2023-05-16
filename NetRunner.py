import torch
import torch.nn as nn
from torchvision import models
from load_stanford_dogs import test_loader, train_loader, val_loader

class NetRunner:
    def __init__(self, train_loader, val_loader, test_loader, num_classes=3):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

        # Load a pre-trained model
        self.model = models.resnet18(pretrained=True)

        # Replace the last layer with a new layer for our specific task
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Move the model to the GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization step
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch: {epoch+1}/{num_epochs} Loss: {running_loss/len(self.train_loader)}")

    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")

    def predict(self, images):
        # Move the images to the GPU if available
        images = images.to(self.device)

        # Forward pass
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        return predicted

net_runner = NetRunner(train_loader, val_loader, test_loader)
net_runner.train()
net_runner.evaluate()

# Get a batch of images from the test loader
images, labels = next(iter(test_loader))

# Make predictions using the trained model
predicted = net_runner.predict(images)

# Print the true and predicted labels
print(f"True labels: {labels}")
print(f"Predicted labels: {predicted}")

from sklearn.metrics import confusion_matrix
import numpy as np

# Get all the true and predicted labels for the test set
true_labels = []
predicted_labels = []
for data in test_loader:
    images, labels = data
    predicted = net_runner.predict(images)
    true_labels.extend(labels.tolist())
    predicted_labels.extend(predicted.tolist())

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Print the confusion matrix
print(cm)

import matplotlib.pyplot as plt

import numpy as np

# Get all the true and predicted labels for the test set
true_labels = []
predicted_labels = []
misclassified_images = []
for data in test_loader:
    images, labels = data
    predicted = net_runner.predict(images)
    true_labels.extend(labels.tolist())
    predicted_labels.extend(predicted.tolist())
    
    # Find the misclassified images
    mask = predicted != labels
    misclassified_images.extend(images[mask].tolist())

# Plot the misclassified images
num_images = len(misclassified_images)
fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
for i in range(num_images):
    # Convert the image to a NumPy array
    image_array = np.array(misclassified_images[i])
    
    # Transpose the array
    transposed_image = image_array.transpose((1, 2, 0))
    
    # Display the transposed image
    axs[i].imshow(transposed_image)
    axs[i].axis('off')  
plt.show()