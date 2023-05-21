from user_interface import UserInterface
from data_model_manager import DataModelManager
from dataloader import DogBreedDataset
from datatransform import DataTransform
from torch.utils.data import DataLoader
ò
def main():
    root_dir = r'StanfordDogs'
    ui = UserInterface()
    dm = DataModelManager(root_dir)

    train = ui.ask_train()
    if train:
        preview = ui.ask_preview()
        early_stopping_patience = ui.ask_early_stopping_patience()
        data_transform = DataTransform(augment=False)
        validation_dataset = DogBreedDataset(root_dir=root_dir, transform=data_transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32)
        ui.show_loading()
        dm.train_model(preview, validation_dataloader, early_stopping_patience)
        ui.hide_loading()

    evaluate = ui.ask_evaluate()
    if evaluate:
        dm.evaluate_model()

    img_path = ui.ask_image_path()
    breed = dm.predict_breed(img_path)
    ui.show_prediction(breed)

if __name__ == '__main__':
    main()
