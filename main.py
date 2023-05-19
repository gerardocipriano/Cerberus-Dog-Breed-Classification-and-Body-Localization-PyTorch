import torch
import tkinter as tk
from tkinter import filedialog
from netrunner import NetRunner
from prediction import Prediction

def main():
    root = tk.Tk()
    root.iconbitmap(r'res/logo.ico')
    root.withdraw()
    root_dir = r'StanfordDogs'
    train = tk.messagebox.askyesno('Train', 'Do you want to train the model?')
    preview = tk.messagebox.askyesno('Preview', 'Do you want to preview the first batch of training?')
    netrunner = NetRunner(root_dir=root_dir, train=train, preview=preview)
    torch.save(netrunner.model, 'model.pth')
    prediction = Prediction(model_path='model.pth')
    img_path = filedialog.askopenfilename(filetypes=[('JPG files', '*.jpg')])
    breed = prediction.predict(img_path)
    tk.messagebox.showinfo('Prediction', f'The predicted breed is: {breed}')

if __name__ == '__main__':
    main()
