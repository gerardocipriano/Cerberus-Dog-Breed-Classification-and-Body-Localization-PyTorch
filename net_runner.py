from datetime import datetime
import shutil
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torchvision
import warnings
from dataloader import DogDataset


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
        name = 'dog_exp_'
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
        hparams_list = [{'learning_rate': 0.001, 'momentum': 0.9, 'batch_size': 4},
                        {'learning_rate': 0.01, 'momentum': 0.9, 'batch_size': 8},
                        {'learning_rate': 0.1, 'momentum': 0.9, 'batch_size': 16}]
        best_acc = 0.0
        best_model_wts = None
        for hparams in hparams_list:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams['learning_rate'], momentum=hparams['momentum'])
            self.train_loader = DataLoader(self.train_set.dataset, batch_size=hparams['batch_size'], shuffle=True)
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
                tag_prefix = f'lr={hparams["learning_rate"]}_bs={hparams["batch_size"]}'
                self.writer.add_scalar(f'Loss/train/{tag_prefix}', epoch_loss, epoch)
                self.writer.add_scalar(f'Accuracy/train/{tag_prefix}', epoch_acc, epoch)

                # Calculate and visualize confusion matrix
                warnings.filterwarnings('ignore')
                cm = confusion_matrix(all_labels, all_preds)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(cm)
                fig.colorbar(cax)
                ax.set_xticklabels([''] + self.train_set.dataset.classes)
                ax.set_yticklabels([''] + self.train_set.dataset.classes)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                self.writer.add_figure(f'confusion_matrix_train/{tag_prefix}', fig, epoch)
                plt.close()

                # Calcolo della confusion matrix per il validation set e della loss di validation per ogni epoca.
                all_labels_valset = []
                all_preds_valset = []
                running_loss_valset = 0.0
                for inputs_valset, labels_valset in self.val_set:
                    inputs_valset = inputs_valset.to(self.device)
                    labels_valset = labels_valset.to(self.device)
                    with torch.set_grad_enabled(False):
                        outputs_valset = self.model(inputs_valset)
                        _, preds_valset = torch.max(outputs_valset, 1)
                        loss_valset = self.criterion(outputs_valset, labels_valset)
                    all_labels_valset.extend(labels_valset.cpu().numpy())
                    all_preds_valset.extend(preds_valset.cpu().numpy())
                    running_loss_valset += loss_valset.item() * inputs_valset.size(0)
                epoch_loss_valset = running_loss_valset / len(self.val_set.dataset)
                print(f'Validation Loss: {epoch_loss_valset:.4f}')
                self.writer.add_scalar(f'Loss/validation/{tag_prefix}', epoch_loss_valset, epoch)

                cm_valset = confusion_matrix(all_labels_valset, all_preds_valset)
                fig_valset_cm = plt.figure()
                ax_valset_cm = fig_valset_cm.add_subplot(111)
                cax_valset_cm=ax_valset_cm.matshow(cm_valset) 
                fig_valset_cm.colorbar(cax_valset_cm) 
                ax_valset_cm.set_xticklabels([''] + self.val_set.dataset.classes) 
                ax_valset_cm.set_yticklabels([''] + self.val_set.dataset.classes) 
                plt.xlabel('Predicted') 
                plt.ylabel('True') 
                self.writer.add_figure(f'confusion_matrix_validation/{tag_prefix}', fig_valset_cm, epoch) 
                plt.close()

                # Add embeddings to TensorBoard
                features = []
                labels = []
                images = []
                for inputs, label in self.val_set:
                    inputs = inputs.to(self.device)
                    with torch.set_grad_enabled(False):
                        output = self.model(inputs)
                    features.append(output)
                    labels.append(label)
                    images.append(inputs)
                features = torch.cat(features).cpu().numpy()
                labels = torch.cat(labels).cpu().numpy()
                images = torch.cat(images).cpu()
                class_names = self.val_set.dataset.classes
                label_names = [class_names[i] for i in labels]
                metadata = [f'{label}:{name}' for label, name in zip(labels, label_names)]
                val_acc = self.evaluate(self.val_set)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_wts = self.model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.config['early_stopping_patience']:
                        break
            self.writer.add_hparams(hparams, {'best_acc': best_acc})
        self.model.load_state_dict(best_model_wts)
        save_path = self.model_path
        torch.save(best_model_wts, save_path)


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
