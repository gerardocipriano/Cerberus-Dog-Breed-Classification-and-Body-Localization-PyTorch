import torch
from dataloader import DogBreedDataset
from datatransform import DataTransform
from netrunner import NetRunner
from prediction import Prediction
from torch.utils.data import DataLoader

class DataModelManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def train_model(self, preview, validation_dataloader, early_stopping_patience=3):
        netrunner = NetRunner(root_dir=self.root_dir, train=True, preview=preview)
        netrunner.train(preview, validation_dataloader, early_stopping_patience)
        torch.save(netrunner.model, 'model.pth')

    def evaluate_model(self):
        data_transform = DataTransform(augment=False)
        validation_dataset = DogBreedDataset(root_dir=self.root_dir, transform=data_transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32)
        netrunner = NetRunner(root_dir=self.root_dir, train=False, preview=False)
        netrunner.evaluate(validation_dataloader)

    def predict_breed(self, img_path):
        prediction = Prediction(model_path='model.pth')
        return prediction.predict(img_path)
