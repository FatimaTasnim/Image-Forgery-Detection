import numpy as np
import matplotlib as plt
import pandas as pd
import torch 
import os
import glob
import cv2
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
from PIL import Image

src_path = "C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/C4_flickr"
dst_path = "C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/SelectedImages/"
input_path = "Image_Forged/benchmark_data/SelectedImages"


# Image Preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((512,512)), # keeping the real size of the image
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

# Train Data
class ForgedDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        #print(self.data)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print("go ", self.data.loc[idx, 'Images'])
        img_name = (self.path + '/' + self.data.iloc[idx, 1])
        #print(img_name)
        img = cv2.imread(img_name)
        img = img.transpose((2, 0, 1))
        #img = torch.from_numpy(img)
        #print("Length ", len(img))
      
        labels = self.data.iloc[idx, 2]
       
        labels = np.array([labels])
        #labels = torch.from_numpy(labels).view(1, -1).type(torch.LongTensor)
        labels = labels.reshape(-1, 1)
        #print("Labels ", labels, type(labels))
        sample = {'image': img, 'labels': labels}
        return sample

# Loading Train Data
dir_train_img = "Image_Forged/benchmark_data/SelectedImages"
train_dataset = ForgedDataset(
    csv_file='C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/classifications_v1.csv', path=dir_train_img, transform=transforms, labels=True)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                          shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The Resnet50 Model
model = models.resnet50(pretrained=True).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.Sigmoid(),
               nn.Linear(128, 2)).to(device) # as the final output will be either 1 or 0

# Loss Function and Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Training the model for 30 epochs
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        

        running_loss = 0.0
        running_corrects = 0

        for step, batch in enumerate(data_loader_train):
            inputs = batch["image"]
            labels = batch["labels"].view(-1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            #print("Outputs: ", outputs)
            #print("preds: ", preds)
            #print("labels.data: ", labels.data, torch.sum(preds == labels.data))
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print(running_corrects)
           
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return model


model_trained = train_model(model, criterion, optimizer, num_epochs=30)

model.eval()
torch.save(model_trained.state_dict(), "resnet50.pth")