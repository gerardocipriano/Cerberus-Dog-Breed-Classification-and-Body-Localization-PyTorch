import subprocess
import tkinter as tk
from tkinter import filedialog
import os
from tkinter import messagebox
from PIL import Image, ImageTk

class UserInterface:
    def __init__(self, model):
        self.model= model
        self.window = tk.Tk()
        self.window.title('Cerberus - Bodyparts Detector')
        self.window.geometry('1440x900')
        self.window.iconbitmap('res/logo.ico')
        self.weights_path = None
        self.train_button_clicked = False
        self.images_path = ''

        image = tk.PhotoImage(file='res/cerberus.png')
        image = image.subsample(3, 3)
        image_label = tk.Label(self.window, image=image)
        image_label.image = image
        image_label.pack()
    
    def run(self):

        self._build_detection_frame()

        self._build_evaluate_frame()
        
        self._build_text_widget()

        self.window.mainloop()


    def _build_model_selection_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        
        label = tk.Label(frame, text='Select model:')
        label.pack(side=tk.LEFT, padx=10)
        
        button = tk.Button(frame, text='Browse', command=self._select_model)
        button.pack(side=tk.RIGHT, padx=10)
        

    def _build_detection_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        
        button = tk.Button(frame, text='Upload image', command=self._upload_image)
        button.pack(side=tk.LEFT, padx=10)


        button = tk.Button(frame, text='Detect', command=self._detect)
        button.pack(side=tk.RIGHT, padx=10)

    def _build_evaluate_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        
        button = tk.Button(frame, text='Show model validation data', command=self._evaluate_model)
        button.pack(side=tk.RIGHT, padx=10)

    def _build_text_widget(self):
        self.text_widget = tk.Text(self.window, height=25, width=100)
        self.text_widget.pack()


    def _select_model(self):
        self.weights_path = filedialog.askopenfilename(filetypes=[('PyTorch model', '*.pt')])
        
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Invalid path")
            return
        
        self.model.set_model(self.weights_path)
        print(f'Selected model: {self.weights_path}')


    def _upload_image(self):
        self.images_path=filedialog.askopenfilename(filetypes=[('Image', '*.jpg;*.jpeg;*.png')])
        print(f'Detection of: {self.images_path}')


    def _detect(self):
        command = [
            'python', './yolov5/detect.py',
            '--weights', self.model.get_weights(),
            '--source', self.images_path,
            '--data', self.model.get_data(),
        ]
        
        try:
            subprocess.run(command, check=True)
            
            
            output_directory = './yolov5/runs/detect'  # Specify the path to the detection results directory
            latest_directory = max(
                [os.path.join(output_directory, d) for d in os.listdir(output_directory)],
                key=os.path.getmtime
            )

            # Get the path to the output image within the latest directory
            output_images = [f for f in os.listdir(latest_directory)]
            
            if len(output_images) > 0:
                output_image_path = os.path.join(latest_directory, output_images[0])
    
            # Load the output image using PIL
            output_image = Image.open(output_image_path)
            
            # Create a new window to display the image
            window = tk.Toplevel(self.window)
            window.title("Detection Output")
            
            # Create a Tkinter compatible image from the PIL image
            image_tk = ImageTk.PhotoImage(output_image)
            
            # Create a label widget to display the image
            image_label = tk.Label(window, image=image_tk)
            image_label.image = image_tk
            image_label.pack()
            
            # Update the Tkinter event loop
            self.window.update()

        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")


    def _evaluate_model(self):
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, "Running validation.\n")
        self.window.update()
        
        try:
            output = subprocess.check_output(['python', './yolov5/val.py', '--weights', self.model.get_weights(), '--data', './yolov5/Cerberus-12/data.yaml', '--img', str(self.model.get_img_size())], stderr=subprocess.STDOUT, universal_newlines=True)
            self.text_widget.insert(tk.END, output)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Function execution failed with error:\n{e.output}")
    
    