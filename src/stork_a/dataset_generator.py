from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import os

# Function to make dataset object
class DatasetGenerator(Dataset):
    def __init__ (self, input_images, transform):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.input_images = input_images
        self.transform = transform

    def __getitem__(self, index):
        image = self.input_images[index]
        path = os.path.join(image.directory, image.filename)
        image_data = Image.open(path).convert('RGB')
        image_metadata = torch.FloatTensor([image.label] + image.metadata.flat_data())
        if self.transform != None: image_data = self.transform(image_data)

        return image_data, image_metadata, path

    def __len__(self):
        return len(self.input_images)
