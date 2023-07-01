import json
from user_interface import UserInterface
from data_model_manager import DataModelManager

class Main:
    def __init__(self):
        with open('config.json') as f:
            config = json.load(f)
        self.data_model_manager = DataModelManager(config)
        self.ui = UserInterface(self.data_model_manager)

    def run(self):
        self.ui.run()

if __name__ == '__main__':
    main = Main()
    main.run()
