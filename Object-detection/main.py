import json
from user_interface import UserInterface
from model import Model

class Main:
    def __init__(self, model):
        self.ui = UserInterface(model)

    def run(self):
        self.ui.run()

if __name__ == '__main__':

    weights = './Cerberus/training_test_/weights/best.pt'
    conf = 0.5
    img_size = 640
    data = './yolov5/Cerberus-12/data.yaml'

    model = Model(weights, img_size, conf, data)
    main = Main(model)
    main.run()

