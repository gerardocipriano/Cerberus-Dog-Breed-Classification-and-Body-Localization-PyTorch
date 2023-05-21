import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

class UserInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.iconbitmap(r'res/logo.ico')
        self.root.withdraw()

    def ask_train(self):
        return tk.messagebox.askyesno('Train', 'Do you want to train the model?')

    def ask_preview(self):
        return tk.messagebox.askyesno('Preview', 'Do you want to preview the first batch of training?')

    def show_loading(self):
        self.loading_window = tk.Toplevel()
        loading_label = tk.Label(self.loading_window, text='Training in progress...')
        loading_label.pack()

    def hide_loading(self):
        self.loading_window.destroy()

    def ask_evaluate(self):
        return tk.messagebox.askyesno('Evaluate', 'Do you want to evaluate the model?')

    def show_prediction(self, breed):
        tk.messagebox.showinfo('Prediction', f'The predicted breed is: {breed}')

    def ask_image_path(self):
        return filedialog.askopenfilename(filetypes=[('JPG files', '*.jpg')])

    def ask_early_stopping_patience(self):
        return simpledialog.askinteger('Early Stopping', 'Enter the early stopping patience:', parent=self.root, minvalue=1)
