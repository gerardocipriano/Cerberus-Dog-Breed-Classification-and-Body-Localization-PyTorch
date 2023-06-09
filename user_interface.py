import tkinter as tk
from tkinter import filedialog
import os
from tkinter import messagebox

class UserInterface:
    def __init__(self, data_model_manager):
        self.data_model_manager = data_model_manager
        self.window = tk.Tk()
        self.window.title('Cerberus - Dog Breed Classifier')
        self.window.geometry('500x600')
        self.window.iconbitmap('res/logo.ico')
        self.model_path = None
        self.train_button_clicked = False
        
        image = tk.PhotoImage(file='res/cerberus.png')
        image = image.subsample(3, 3)
        image_label = tk.Label(self.window, image=image)
        image_label.image = image
        image_label.pack()
    
    def run(self):
        self._build_model_selection_frame()
        self._build_train_frame()
        self._build_evaluate_frame()
        self._build_predict_frame()
        
        # Add a new button to train the model on cats
        self._build_cat_train_frame()
        
        self.window.mainloop()
    
    def _build_model_selection_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        
        label = tk.Label(frame, text='Select model:')
        label.pack(side=tk.LEFT, padx=10)
        
        button = tk.Button(frame, text='Browse', command=self._select_model)
        button.pack(side=tk.RIGHT, padx=10)
        
        create_button = tk.Button(frame, text='Create New Model', command=self._create_new_model)
        create_button.pack(side=tk.RIGHT, padx=10)
    
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

    def _build_cat_train_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)

        label = tk.Label(frame, text='Train model on cats:')
        label.pack(side=tk.LEFT, padx=10)

        button = tk.Button(frame, text='Train', command=self._train_cat_model, state=tk.DISABLED)
        button.pack(side=tk.RIGHT, padx=10)

        self.cat_train_button = button

    def _select_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[('PyTorch model', '*.pth')])
        
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "The specified model path is invalid. Please create a new model using the 'Create New Model' button.")
            return
        
        print(f'Selected model: {self.model_path}')
        self.data_model_manager.set_model(self.model_path)
        
    def _create_new_model(self):
       save_path=self.data_model_manager.create_new_model()
       self.model_path=save_path
       self.data_model_manager.set_model(self.model_path)

    def _train_model(self):
       print('Training model on dog breeds...')
       # Set the train_button_clicked flag to True to enable the cat_train_button
       self.train_button_clicked=True
       self.cat_train_button.config(state=tk.NORMAL)

       self.data_model_manager.train_model()

    def _evaluate_model(self):
       print('Evaluating model...')
       self.data_model_manager.test_model()

    def _predict_breed(self):
        image_path = filedialog.askopenfilename(filetypes=[('Image', '*.jpg;*.jpeg;*.png')])
        print(f'Predicting breed for image: {image_path}')
        pred_class = self.data_model_manager.predict_breed(image_path)
        print(f'Predicted breed: {pred_class}')
        
        messagebox.showinfo("Prediction Result", f"The predicted breed is: {pred_class}")


    def _train_cat_model(self):
       print('Training model on cats...')
       # Call the train_cat_model method of the data model manager to train the model on cats
       self.data_model_manager.train_cat_model()

