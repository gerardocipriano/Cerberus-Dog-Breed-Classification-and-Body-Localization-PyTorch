from user_interface import UserInterface
from data_model_manager import DataModelManager
from dataloader import DogBreedDataset
from datatransform import DataTransform
from torch.utils.data import DataLoader

def main():
    root_dir = r'StanfordDogs'
    ui = UserInterface()
    dm = DataModelManager(root_dir)

    # Always ask the user for the model path
    model_path = ui.ask_model_path()

    train = ui.ask_train()
    if train:
        preview = ui.ask_preview()
        early_stopping_patience = ui.ask_early_stopping_patience()
        data_transform = DataTransform(augment=False)
        validation_dataset = DogBreedDataset(root_dir=root_dir, transform=data_transform)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32)
        dm.train_model(preview, validation_dataloader, early_stopping_patience, model_path)

    evaluate = ui.ask_evaluate()
    if evaluate:
        dm.evaluate_model(model_path)

    img_path = ui.ask_image_path()
    breed = dm.predict_breed(img_path, model_path)
    ui.show_prediction(breed)

if __name__ == '__main__':
    main()