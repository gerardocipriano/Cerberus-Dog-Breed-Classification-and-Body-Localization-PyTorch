from dataloader import DogDataset
from net_runner import NetRunner
from prediction import Predictor
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class DataModelManager:
    def __init__(self, config):
        self.config = config
        data_transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = DogDataset(config['root_folder'], dataset_type='train', transform=data_transform)
        self.validation_dataset = DogDataset(config['root_folder'], dataset_type='validation', transform=data_transform)
        self.test_dataset = DogDataset(config['root_folder'], dataset_type='test', transform=data_transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=config['batch_size'], shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config['batch_size'], shuffle=False)
        self.net_runner = None
        self.predictor = None

    def set_model(self, model_path):
        self.net_runner = NetRunner(model_path, self.train_loader, self.test_loader, self.validation_loader, self.config, num_classes=len(self.train_dataset.classes))
        self.predictor = Predictor(model_path, num_classes=len(self.train_dataset.classes), class_names=self.train_dataset.classes)

    def train_model(self):
        if not self.net_runner:
            raise ValueError('Model not set')
        self.net_runner.train()

    def test_model(self):
        if not self.net_runner:
            raise ValueError('Model not set')
        self.net_runner.test()

    def predict_breed(self, image_path):
        if not self.predictor:
            raise ValueError('Model not set')
        pred_class = self.predictor.predict(image_path)
        return pred_class
