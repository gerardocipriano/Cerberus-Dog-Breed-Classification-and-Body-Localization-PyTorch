from user_interface import UserInterface
from data_model_manager import DataModelManager

def main():
    root_dir = r'StanfordDogs'
    ui = UserInterface()
    dm = DataModelManager(root_dir)

    train = ui.ask_train()
    if train:
        preview = ui.ask_preview()
        ui.show_loading()
        dm.train_model(preview)
        ui.hide_loading()

    evaluate = ui.ask_evaluate()
    if evaluate:
        dm.evaluate_model()

    img_path = ui.ask_image_path()
    breed = dm.predict_breed(img_path)
    ui.show_prediction(breed)

if __name__ == '__main__':
    main()
