import torch
import os
import time
from datetime import datetime
from torchvision.models import alexnet
from tkinter import messagebox


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
            print(f'INFO - Loaded model weights from {model_path}')
            return model
        except Exception as e:
            print(f'ERROR - Error loading model from {model_path}: {e}')
    else:
        # Prompt the user that the provided model path is invalid
        messagebox.showerror("Error", "The specified model path is invalid. Please create a new model using the 'Create New Model' button.")
        return None

def create_new_model(num_classes):
    # Create a new default AlexNet model and save it to disk
    print(f'INFO - Creating new default AlexNet model')
    model = alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

    # Save the new AlexNet model to disk with a timestamp in the filename
    timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f'alexnet_pretrained_{date_time}.pth'
    torch.save(model.state_dict(), save_path)

    # Set the newly created AlexNet model as the current model
    print(f'INFO - Saved default AlexNet model to {save_path}')
    return save_path
