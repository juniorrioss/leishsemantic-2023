from glob import glob
import json


# def load_json(path):
#   with open('')


class LeishDataset:
    def __init__(self, folder):
        self.folder = folder
        self.json_list = glob(folder + "/*.json")

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, idx):
        pass


from datasets import Dataset, DatasetDict, Image

# your images can of course have a different extension
# semantic segmentation maps are typically stored in the png format
image_paths_train = glob("data/raw_data/*.png")
label_paths_train = glob("data/label/*")


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict(
        {"pixel_values": sorted(image_paths), "label": sorted(label_paths)}
    )
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset


# step 1: create Dataset objects
train_dataset = create_dataset(image_paths_train, label_paths_train)
# validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# step 2: create DatasetDict
dataset = DatasetDict(
    {
        "train": train_dataset,
        # "validation": val_dataset,
    }
)
