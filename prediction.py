import torch
import os
from PIL import Image
from torchvision.models import alexnet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils import load_alexnet_model

class Predictor:
    def __init__(self, model_path, num_classes, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, num_classes).to(self.device)
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def _load_model(self, model_path, num_classes):
        self.model = load_alexnet_model(model_path, num_classes).to(self.device)
        return self.model




    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.set_grad_enabled(False):
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
        return self.class_names[preds[0]]

