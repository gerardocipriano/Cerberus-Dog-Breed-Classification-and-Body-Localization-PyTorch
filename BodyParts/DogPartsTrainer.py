import os
import torch
import torch.nn as nn
import torch.optim as optim
from DogPartsModel import DogPartsModel
from DogPartsDataset import DogPartsDataset

class DogPartsTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, num_epochs):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0

            for images, annotations in self.train_dataloader:
                images = images.to(self.device)
                annotations = annotations.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                outputs = outputs.squeeze(0)  # Remove the extra batch dimension
                print("Output shape:", outputs.shape)
                print("Annotation shape:", annotations.shape)
                loss = self.criterion(outputs, annotations)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

            # Validation
            if self.val_dataloader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for val_images, val_annotations in self.val_dataloader:
                        val_images = val_images.to(self.device)
                        val_annotations = val_annotations.to(self.device)

                        val_outputs = self.model(val_images)
                        val_outputs = val_outputs.squeeze(0)  # Remove the extra batch dimension
                        val_loss += self.criterion(val_outputs, val_annotations).item()

                    val_loss /= len(self.val_dataloader)
                    print(f"Validation Loss: {val_loss:.4f}")

                self.model.train()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

num_classes = 3
# Define your model
model = DogPartsModel(num_classes)

data_dir = r'BodyParts'
# Define your training and validation dataloaders
train_dataset = DogPartsDataset(os.path.join(data_dir, ''))
val_dataset = DogPartsDataset(os.path.join(data_dir, ''))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define your loss function
criterion = nn.MSELoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs
num_epochs = 3

# Create an instance of the trainer
trainer = DogPartsTrainer(model, train_dataloader, val_dataloader, criterion, optimizer)

# Start the training
trainer.train(num_epochs)

# Save the trained model
trainer.save_model('BodyParts/model.pth')
