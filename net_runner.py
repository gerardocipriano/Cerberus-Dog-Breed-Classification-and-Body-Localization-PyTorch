from datetime import datetime
import shutil
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torchvision

from utils import load_alexnet_model

class NetRunner:
    def __init__(self, model_path, train_set, test_set, val_set, config, num_classes):
        # Remove all files and subfolders in the 'runs' folder
        shutil.rmtree('runs', ignore_errors=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.config = config
        
        # Set the model_path attribute with the provided model path
        self.model_path = model_path
        
        self.model = self._load_model(model_path, num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

        timestamp = time.time()
        date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
        name = 'cerb_exp_'
        run_name = name + date_time
        self.writer = SummaryWriter(f'runs/{run_name}')

        # Add model graph to TensorBoard
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
        self.writer.flush()

    def _load_model(self, model_path, num_classes):
        self.model = load_alexnet_model(model_path, num_classes).to(self.device)
        return self.model

    def train(self):
        best_acc = 0.0
        early_stopping_counter = 0
        for epoch in range(self.config['num_epochs']):
            print(f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            all_labels = []
            all_preds = []
            for inputs, labels in self.train_set:

                # Visualize the first batch of training images
                img_grid = torchvision.utils.make_grid(inputs)
                self.writer.add_image('first_batch', img_grid, global_step=epoch)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.train_set.dataset)
            epoch_acc = running_corrects.double() / len(self.train_set.dataset)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            # Calculate and visualize confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax=ax.matshow(cm)
            fig.colorbar(cax)
            ax.set_xticklabels([''] + self.train_set.dataset.classes)
            ax.set_yticklabels([''] + self.train_set.dataset.classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            self.writer.add_figure('confusion_matrix_train', fig, epoch)
            plt.close()

            val_acc=self.evaluate(self.val_set)
            if val_acc > best_acc:
                best_acc=val_acc
                best_model_wts=self.model.state_dict()
                
                # Save the best model weights to the specified model path
                save_path = self.model_path
                torch.save(best_model_wts, save_path)
                print(f'Saved best model weights to {save_path}')
                
                early_stopping_counter=0
                print(f'Best val Acc: {best_acc:.4f}')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config['early_stopping_patience']:
                    print(f'Early stopping after {early_stopping_counter} epochs with no improvement')
                    break

        self.model.load_state_dict(best_model_wts)


    def evaluate(self, dataset):
        self.model.eval()
        running_corrects=0
        for inputs, labels in dataset:
            inputs=inputs.to(self.device)
            labels=labels.to(self.device)
            with torch.set_grad_enabled(False):
                outputs=self.model(inputs)
                _, preds=torch.max(outputs, 1)
                running_corrects+=torch.sum(preds==labels.data)
        acc=running_corrects.double()/len(dataset.dataset)
        return acc

    def test(self):
        acc=self.evaluate(self.test_set)
        print(f'Test Acc: {acc:.4f}')
