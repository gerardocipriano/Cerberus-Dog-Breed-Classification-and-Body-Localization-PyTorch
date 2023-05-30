import tkinter as tk
from tkinter import filedialog

class UserInterface:
    def __init__(self, data_model_manager):
        self.data_model_manager = data_model_manager
        self.window = tk.Tk()
        self.window.title('Cerberus - Dog Breed Classifier')
        self.window.geometry('400x200')
        self.window.iconbitmap('res/logo.ico')
        self.model_path = None

    def run(self):
        self._build_model_selection_frame()
        self._build_train_frame()
        self._build_evaluate_frame()
        self._build_predict_frame()
        self.window.mainloop()

    def _build_model_selection_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        label = tk.Label(frame, text='Select model:')
        label.pack(side=tk.LEFT, padx=10)
        button = tk.Button(frame, text='Browse', command=self._select_model)
        button.pack(side=tk.RIGHT, padx=10)

    def _build_train_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        label = tk.Label(frame, text='Train model:')
        label.pack(side=tk.LEFT, padx=10)
        button = tk.Button(frame, text='Train', command=self._train_model)
        button.pack(side=tk.RIGHT, padx=10)

    def _build_evaluate_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        label = tk.Label(frame, text='Evaluate model:')
        label.pack(side=tk.LEFT, padx=10)
        button = tk.Button(frame, text='Evaluate', command=self._evaluate_model)
        button.pack(side=tk.RIGHT, padx=10)

    def _build_predict_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        label = tk.Label(frame, text='Predict dog breed:')
        label.pack(side=tk.LEFT, padx=10)
        button = tk.Button(frame, text='Predict', command=self._predict_breed)
        button.pack(side=tk.RIGHT, padx=10)

    def _select_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[('PyTorch model', '*.pth')])
        print(f'Selected model: {self.model_path}')
        self.data_model_manager.set_model(self.model_path)



    def _train_model(self):
        print('Training model...')
        self.data_model_manager.train_model()



    def _evaluate_model(self):
        print('Evaluating model...')
        self.data_model_manager.test_model()



    def _predict_breed(self):
        image_path = filedialog.askopenfilename(filetypes=[('Image', '*.jpg;*.jpeg;*.png')])
        print(f'Predicting breed for image: {image_path}')
        pred_class = self.data_model_manager.predict_breed(image_path)
        print(f'Predicted breed: {pred_class}')
