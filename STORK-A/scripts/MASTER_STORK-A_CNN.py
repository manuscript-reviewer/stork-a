from __future__ import print_function, division
import psutil
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import roc_curve, auc, roc_auc_score

print('NOW STARTING')

#quick path change
path = '~/Desktop/Github/Data'
quick = os.path.join(path, 'Anueploid vs Euploid/img_110_ANU-EUP') #change path depending on which inputs you want to use for train, validaiton, test

#data directory
data_dir = os.path.join(path, 'images')
train_dir = os.path.join(quick, 'training.txt')
val_dir = os.path.join(quick, 'validation.txt')
test_dir = os.path.join(quick, 'test.txt')

#load input to get number of clincal features and class weight sizes
get_vals = pd.read_csv(train_dir, sep="\t", header=0)
Total_1 = get_vals['Label'].sum() #normal
Total_0 =  get_vals.shape[0]-Total_1   #abnormal

#parameter tuning
batch_size = 32
size_fc1 = 20
num_clin = get_vals.shape[1] - 3 #number of clinical features
size_fc2 = size_fc1 + num_clin
num_epochs = 20
learn_rate = 1e-4
step_size = 5
Weight_Decay = 0
weights = [1/Total_0, 1/Total_1]
torch.set_printoptions(precision=15)

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.05,0.05,0.05,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Function to make dataset object

class DatasetGenerator (Dataset):
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        fileDescriptor = open(pathDatasetFile, "r")
        line = True
        line = fileDescriptor.readline()
        while line:

            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[1])
                imageLabel = lineItems[2:]
                imageLabel = [float(i) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

        fileDescriptor.close()

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])

        if self.transform != None: imageData = self.transform(imageData)

        return imageData, imageLabel , imagePath

    def __len__(self):

        return len(self.listImagePaths)


# Create an image dataset
image_datasets = {'train': DatasetGenerator(data_dir, train_dir , data_transforms['train']),
                  'val': DatasetGenerator(data_dir, val_dir, data_transforms['val']),
                  'test': DatasetGenerator(data_dir, test_dir, data_transforms['test'])}

# Create an loader to load image into GPU
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False) for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer,  num_epochs=25):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, paths in dataloaders[phase]:

                inputs = inputs.to(device)
                seq = labels[:,1:]
                seq = seq.to(device)
                labels = labels[:,0]
                labels = labels.type('torch.LongTensor')
                labels = labels.to(device)
                # zero the parameter gradients
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs,seq)
                    if isinstance(outputs, tuple):
                        outputs,aux_outputs = outputs
                        #_, preds = torch.max(outputs.data, 1)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Define Deep Learning Architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
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

model_ft = MyModel()
model_ft = model_ft.to(device)

# define loss function and optimizer
class_weights = torch.FloatTensor(weights).cuda()
#class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer_ft = optim.Adam(model_ft.parameters(), lr=learn_rate, weight_decay=Weight_Decay)
exp_LR = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)

# ^^^^^^^^^^^^^^^^^^
# Train and evaluate model
# ^^^^^^^^^^^^^^^^^^
model_ft, hist = train_model(model_ft, dataloaders, criterion,  optimizer_ft,
                       num_epochs=num_epochs)

saveit = os.path.join(path, quick, 'saved_model.pth')
torch.save(model_ft.state_dict(), saveit)

##############################################################################################################################################################

class_prob = []
labs = []
predic = []
IDs = []

model_y = model_ft
model_y.eval()

with torch.no_grad():
    for inputs, labels_test, paths in dataloaders['test']:
        inputs = inputs.to(device)
        seq = labels_test[:, 1:]
        seq = seq.to(device)
        labels_test = labels_test[:, 0]
        labels_test = labels_test.to(device)
        output = model_y(inputs, seq)
        _, predicted = torch.max(output.data, 1)
        output.cpu().numpy()
        prediction = output.softmax(dim=1)
        predic.append(predicted)
        class_prob.append(prediction)
        labs.append(labels_test)
        IDs.extend(paths)


labs = torch.cat(labs).type('torch.LongTensor')
confidence = torch.cat(class_prob)
predicted = torch.cat(predic).type('torch.LongTensor')

#compute accuracy and AUC
total = labs.shape[0]
correct = (predicted == labs).sum().item()
acc = correct / total
print('Final Accuracy: {:4f}'.format(acc))

fpr, tpr, _ = roc_curve(labs.cpu().numpy(), confidence[:,1].cpu().numpy())
roc_auc = auc(fpr, tpr)
print('Final AUC: {:4f}'.format(roc_auc))

labs = pd.DataFrame(labs.tolist())
predicted = pd.DataFrame(predicted.tolist())
confidence = pd.DataFrame(confidence.tolist())
IDs = pd.Series(IDs).str.split('/', expand=True)[6]
IDs = IDs.str.split('_', expand=True)[0]
result = pd.concat([labs, predicted, confidence, IDs], axis=1, sort=False)
result.to_csv(quick+r'/result.csv', index = False,  header=False)