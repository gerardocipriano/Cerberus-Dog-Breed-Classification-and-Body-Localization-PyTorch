import subprocess

class Model:
    def __init__(self, weights, img_size, conf, data):
        self.__weights = weights
        self.__img_size = img_size
        self.__conf = conf
        self.__data = data

    def get_weights(self):
        return self.__weights
    
    def get_img_size(self):
        return self.__img_size
    
    def get_conf(self):
        return self.__conf
    
    def get_data(self):
        return self.__data
    
    def set_model(self, weights):

        self.weights = weights
