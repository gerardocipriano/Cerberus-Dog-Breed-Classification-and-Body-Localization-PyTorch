import torch
import tkinter as tk
from tkinter import filedialog
from dataloader import DogBreedDataset
from netrunner import NetRunner
from prediction import Prediction
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    root = tk.Tk()
    root.iconbitmap(r'res/logo.ico')
    root.withdraw()
    root_dir = r'StanfordDogs'
    train = tk.messagebox.askyesno('Train', 'Do you want to train the model?')
    preview = False
    if train:
        preview = tk.messagebox.askyesno('Preview', 'Do you want to preview the first batch of training?')
        loading_window = tk.Toplevel()
        loading_label = tk.Label(loading_window, text='Training in progress...')
        loading_label.pack()
        netrunner = NetRunner(root_dir=root_dir, train=train, preview=preview)
        loading_window.destroy()
    else:
        netrunner = NetRunner(root_dir=root_dir, train=train, preview=preview)
    evaluate = tk.messagebox.askyesno('Evaluate', 'Do you want to evaluate the model?')
    if evaluate:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        validation_dataset = DogBreedDataset(root_dir=root_dir, transform=transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=4)
        netrunner.evaluate(validation_dataloader)
    torch.save(netrunner.model, 'model.pth')
    prediction = Prediction(model_path='model.pth')
    img_path = filedialog.askopenfilename(filetypes=[('JPG files', '*.jpg')])
    breed = prediction.predict(img_path)
    tk.messagebox.showinfo('Prediction', f'The predicted breed is: {breed}')

if __name__ == '__main__':
    main()

