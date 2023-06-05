from tkinter import messagebox
import torch
import os
from torchvision.models import alexnet

def load_alexnet_model(model_path, num_classes):
    # Check if the provided model path is valid
    if model_path and os.path.exists(model_path):
        try:
            # Create a new AlexNet model
            model = alexnet(weights='DEFAULT')
            # Freeze all the weights in the model
            for param in model.parameters():
                param.requires_grad = False
            num_ftrs = model.classifier[6].in_features
            # Replace the last layer with a new one with the correct number of classes
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
            # Load the saved weights into the model
            model.load_state_dict(torch.load(model_path))
            print(f'Loaded model weights from {model_path}')
            return model
        except Exception as e:
            print(f'Error loading model from {model_path}: {e}')
    else:
        # Prompt the user that the provided model path is invalid
        messagebox.showerror("Error", "The specified model path is invalid. Please create a new model using the 'Create New Model' button.")
        return None
