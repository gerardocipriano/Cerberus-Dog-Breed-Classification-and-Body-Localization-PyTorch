import torch
import os
from PIL import Image
from torchvision.models import alexnet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class Predictor:
    def __init__(self, model_path, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, num_classes).to(self.device)
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path, num_classes):
        if model_path:
            try:
                # Create a new AlexNet model
                model = alexnet(weights='DEFAULT')
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
                # Load the saved weights into the model
                model.load_state_dict(torch.load(model_path))
                print(f'Loaded model weights from {model_path}')
            except Exception as e:
                print(f'Error loading model from {model_path}: {e}')
                print(f'Creating new AlexNet model')
                model = alexnet(weights='DEFAULT')
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
                # Save the downloaded model weights to disk
                save_path = os.path.join(self.config['root_folder'], 'alexnet_pretrained.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Saved downloaded AlexNet model weights to {save_path}')
        else:
            print(f'Creating new AlexNet model')
            model = alexnet(weights='DEFAULT')
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
            # Save the downloaded model weights to disk
            save_path = os.path.join(self.config['root_folder'], 'alexnet_pretrained.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Saved downloaded AlexNet model weights to {save_path}')
        return model

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.set_grad_enabled(False):
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
        return preds[0]
