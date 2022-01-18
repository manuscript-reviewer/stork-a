import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define Deep Learning Architecture
class MyModel(nn.Module):
    def __init__(self, input_images):
        super(MyModel, self).__init__()

        #load input to get number of clinical features and class weight sizes
        #parameter tuning
        num_clin = input_images[0].metadata.get_flat_length()
        size_fc1 = 20
        size_fc2 = size_fc1 + num_clin

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features,size_fc1)
        self.fc1 = nn.Linear(size_fc1 + num_clin , size_fc2)
        self.fc2 = nn.Linear(size_fc2,2)

    def forward(self, image, data):
        x1 = self.cnn(image)
        if isinstance(x1, tuple):
            x1 = x1[0]
        x2 = data
        x = torch.cat((x1,x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
