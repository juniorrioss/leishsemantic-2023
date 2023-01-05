import numpy as np
from PIL import Image
import requests
import torch
from torch import nn
import matplotlib.pyplot as plt


def paleta():
    return [[255, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0]]


from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation

model_name = "models/overfit_weightned_5_moresamples/checkpoint-176"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)


# path = "data/raw_data/2.png"
# image = Image.open(path)
from datasets import Dataset, DatasetDict, Image
from glob import glob

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

image = train_dataset[6]["pixel_values"]
# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# rescale logits to original image size
logits = nn.functional.interpolate(
    outputs.logits.detach().cpu(),
    size=image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False,
)
seg = logits.argmax(1)[0]

color_seg = np.zeros(
    (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
)  # height, width, 3
palette = np.array(paleta())
for label, color in enumerate(palette):

    color_seg[seg == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)

l = logits.softmax(dim=1)
leish_logits = l[0, 1]
leish_s = torch.sigmoid(leish_logits)
leish = leish_s > 0.6
leish = torch.stack([leish, leish, leish])
leish = np.moveaxis(leish.numpy(), 0, 2)
leish = leish.astype(np.uint8) * 255

plt.figure(figsize=(15, 10))
plt.imshow(leish_s)

img = np.array(image) * 0.5 + leish * 0.5
img = img.astype(np.uint8)


plt.figure(figsize=(15, 10))
plt.imshow(img)


# path = "data/label/0.png"
# image = Image.open(path)


plt.figure(figsize=(15, 10))
plt.imshow(image)
plt.show()
print("a")
