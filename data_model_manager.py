import torch
from dataloader import DogBreedDataset
from datatransform import DataTransform
from net_runner import NetRunner
from prediction import Prediction
from torch.utils.data import DataLoader

class DataModelManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def train_model(self, preview, validation_dataloader, early_stopping_patience=3, model_path=None):
        # Create an instance of the DataTransform class with data augmentation enabled
        data_transform = DataTransform(augment=True)

        # Create a training dataset with data augmentation
        training_dataset = DogBreedDataset(root_dir=self.root_dir, transform=data_transform)

        # Create a data loader with the training dataset
        batch_size = 32
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        # Create an instance of the NetRunner class and train the model
        netrunner = NetRunner(root_dir=self.root_dir, train=True, preview=preview, model_path=model_path)
        netrunner.train(training_dataloader, validation_dataloader, early_stopping_patience)


    def evaluate_model(self, model_path=None):
        data_transform = DataTransform(augment=False)
        validation_dataset = DogBreedDataset(root_dir=self.root_dir, transform=data_transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32)
        netrunner = NetRunner(root_dir=self.root_dir, train=False, preview=False, model_path=model_path)
        netrunner.evaluate(validation_dataloader)

    def predict_breed(self, img_path, model_path=None):
        prediction = Prediction(model_path=model_path)
        return prediction.predict(img_path)
