from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import models

def imshow(imgs, labels):
    fig = plt.figure(figsize=(10, 10))
    for i in range(imgs.size(0)):
        img = imgs[i] / 2 + 0.5
        npimg = img.numpy()
        ax = fig.add_subplot(8, 4, i + 1)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(labels[i])
    plt.show()

from torch.optim.lr_scheduler import ReduceLROnPlateau

class NetRunner:
    def __init__(self, root_dir, train=True, preview=True, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.alexnet(weights='DEFAULT' if model_path is None else None)
        self.model.classifier[6] = nn.Linear(4096, 3)
        if model_path is not None:
            print(f'Loading model from {model_path}')
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1E-5, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)

        self.breeds = ['n02088364-beagle', 'n02110185-Siberian_husky', 'n02113624-toy_poodle']

    def train(self, training_dataloader, validation_dataloader, preview=False, early_stopping_patience=3):
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(training_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels_idx = torch.tensor([self.breeds.index(label) for label in labels]).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_idx)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}')
                    running_loss = 0.0
                if preview:
                    predictions = [self.breeds[output.argmax()] for output in outputs]
                    imshow(inputs.cpu(), predictions)
                    preview = False

            # Calculate validation loss
            val_loss = 0.0
            with torch.no_grad():
                for data in validation_dataloader:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels_idx = torch.tensor([self.breeds.index(label) for label in labels]).to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels_idx)
                    val_loss += loss.item()
                val_loss /= len(validation_dataloader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        print('Finished Training')

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels_idx = torch.tensor([self.breeds.index(label) for label in labels]).to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += (predicted == labels_idx).sum().item()
                all_labels.extend(labels)
                all_predictions.extend([self.breeds[prediction] for prediction in predicted])
            accuracy = 100 * correct / len(dataloader.dataset)
            print(f'Accuracy: {accuracy}%')
            cm = confusion_matrix(all_labels, all_predictions, labels=self.breeds)
            print(f'Confusion Matrix:\n{cm}')
