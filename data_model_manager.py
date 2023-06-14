from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from dataloader import DogDataset
from dataloader import CatDataset
from net_runner import NetRunner
from cat_trainer import CatTrainer
from prediction import Predictor
from utils import create_new_model

class DataModelManager:
    def __init__(self, config):
        self.config = config
        self._initialize_data_loaders()
        self.net_runner = None
        self.predictor = None
        self.cat_trainer = None

    def set_model(self, model_path):
        num_classes = len(self.train_dataset.classes)
        self.net_runner = NetRunner(model_path, self.train_loader, self.test_loader, self.validation_loader, self.config, num_classes)
        self.predictor = Predictor(model_path, num_classes, class_names=self.train_dataset.classes)

    def train_model(self):
        if not self.net_runner:
            raise ValueError('Model not set')
        self.net_runner.train()
        num_classes = len(self.train_dataset.classes)
        self.predictor = Predictor(self.net_runner.model_path, num_classes, class_names=self.train_dataset.classes)
        
        # Update the cat_trainer with the trained model
        if self.cat_trainer:
            self.cat_trainer.set_model(self.net_runner.model_path)

    def test_model(self):
        if not self.net_runner:
            raise ValueError('Model not set')
        self.net_runner.test()

    def predict_breed(self, image_path):
        if not self.predictor:
            raise ValueError('Model not set')
        pred_class = self.predictor.predict(image_path)
        return pred_class

    def create_new_model(self):
        num_classes = len(self.train_dataset.classes)
        save_path = create_new_model(num_classes)
        return save_path

    def _initialize_data_loaders(self):
        data_transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        root_folder = self.config['root_folder']
        batch_size = self.config['batch_size']
        
        # Initialize the dog data loaders
        self.train_dataset = DogDataset(root_folder, dataset_type='train', transform=data_transform)
        self.validation_dataset = DogDataset(root_folder, dataset_type='validation', transform=data_transform)
        self.test_dataset = DogDataset(root_folder, dataset_type='test', transform=data_transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the cat data loaders
        self.cat_train_dataset = CatDataset(root_folder, dataset_type='train', transform=data_transform)
        self.cat_validation_dataset = CatDataset(root_folder, dataset_type='validation', transform=data_transform)
        self.cat_test_dataset = CatDataset(root_folder, dataset_type='test', transform=data_transform)
        
        self.cat_train_loader = DataLoader(self.cat_train_dataset, batch_size=batch_size, shuffle=True)
        self.cat_validation_loader = DataLoader(self.cat_validation_dataset, batch_size=batch_size, shuffle=False)
    
    def train_cat_model(self):

       if not self.cat_trainer:
           num_classes=len(self.cat_train_dataset.classes)
           cat_config=self.config.copy()
           cat_config['num_epochs']=5
           cat_config['batch_size']=4
           cat_config['early_stopping_patience']=2
           cat_config['learning_rate']=0.001

           # Create a new instance of the CatTrainer class and pass it the trained model path and the cat data loaders
           model_path=self.net_runner.model_path
           train_set=self.cat_train_loader
           val_set=self.cat_validation_loader

           # Create a new instance of the CatTrainer class and pass it the trained model path and the cat data loaders
           model_path=self.net_runner.model_path
           train_set=self.cat_train_loader
           val_set=self.cat_validation_loader

           # Create a new instance of the CatTrainer class and pass it the trained model path and the cat data loaders
           model_path=self.net_runner.model_path
           train_set=self.cat_train_loader
           val_set=self.cat_validation_loader

           self.cat_trainer=CatTrainer(model_path, train_set, val_set, cat_config, num_classes)

       self.cat_trainer.train()
