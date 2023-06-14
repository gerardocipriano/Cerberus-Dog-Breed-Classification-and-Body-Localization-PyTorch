from datetime import datetime
import time
import torch
from utils import load_alexnet_model
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import warnings

class CatTrainer:
    def __init__(self, model_path, train_set, val_set, config, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_set = train_set
        self.val_set = val_set
        self.config = config
        self.model_path = model_path
        self.model = self._load_model(model_path, num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
        timestamp = time.time()
        date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
        name = 'cat_exp_'
        run_name = name + date_time
        self.writer = SummaryWriter(f'runs/{run_name}')
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
        self.writer.flush()

    def _load_model(self, model_path, num_classes):
        self.model = load_alexnet_model(model_path, num_classes).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        return self.model

    def train(self):
        best_acc = 0.0
        early_stopping_counter = 0
        for epoch in range(self.config['num_epochs_cats']):
            print(f'Epoch {epoch + 1}/{self.config["num_epochs_cats"]}')
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            for i, (inputs, labels) in enumerate(self.train_set):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i == 0:
                    img_grid = torchvision.utils.make_grid(inputs)
                    self.writer.add_image('first_batch', img_grid, epoch)
            epoch_loss = running_loss / len(self.train_set.dataset)
            epoch_acc = running_corrects.double() / len(self.train_set.dataset)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            val_acc, cm = self.evaluate(self.val_set)
            # Calculate and visualize confusion matrix
            warnings.filterwarnings('ignore')
            fig=plt.figure()
            ax=fig.add_subplot(111)
            cax=ax.matshow(cm,cmap='Blues')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + self.train_set.dataset.classes)
            ax.set_yticklabels([''] + self.train_set.dataset.classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            self.writer.add_figure('confusion_matrix_train', fig, epoch)
            plt.close()

            if val_acc > best_acc:
                best_acc=val_acc
                print(f'CAT - TRAINING - INFO - Best val Acc: {best_acc:.4f}')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 3:
                    print(f'CAT - TRAINING - INFO - Early stopping after {early_stopping_counter} epochs with no improvement')
                    break


        print('CAT - TRAINING - INFO - Training finished')

    def evaluate(self, dataset):
        self.model.eval()
        all_preds = []
        all_labels = []
        for inputs, labels in dataset:
            inputs=inputs.to(self.device)
            labels=labels.to(self.device)
            with torch.set_grad_enabled(False):
                outputs=self.model(inputs)
                _, preds=torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm=confusion_matrix(all_labels,all_preds)
        acc=np.sum(np.diag(cm))/np.sum(cm)
        return acc, cm
