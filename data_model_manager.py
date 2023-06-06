
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from dataloader import DogDataset
from net_runner import NetRunner
from prediction import Predictor

from utils import create_new_model

class DataModelManager:
    def __init__(self, config):
        self.config = config
        self._initialize_data_loaders()
        self.net_runner = None
        self.predictor = None

    def set_model(self, model_path):
        num_classes = len(self.train_dataset.classes)
        self.net_runner = NetRunner(model_path, self.train_loader, self.test_loader, self.validation_loader, self.config, num_classes)
        self.predictor = Predictor(model_path, num_classes, class_names=self.train_dataset.classes)

    def train_model(self):
        if not self.net_runner:
            raise ValueError('Model not set')
        
        # Train the model
        self.net_runner.train()
        
        # Update the Predictor with the new model weights
        num_classes = len(self.train_dataset.classes)
        self.predictor = Predictor(self.net_runner.model_path, num_classes, class_names=self.train_dataset.classes)

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
        
        self.train_dataset = DogDataset(root_folder, dataset_type='train', transform=data_transform)
        self.validation_dataset = DogDataset(root_folder, dataset_type='validation', transform=data_transform)
        self.test_dataset = DogDataset(root_folder, dataset_type='test', transform=data_transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)