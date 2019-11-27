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

# Prepared Dataset Can be Downloaded from https://drive.google.com/open?id=1wlJukwPmk3yAVzELeCHL6KHfoiuKUieK
# Data Paths 
src_path = "C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/C4_flickr"
dst_path = "C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/Image_Forged/benchmark_data/SelectedImages/"

# Image Preprocessing
# As it's a starter code so only using basic stuffs
transforms = transforms.Compose(
    [transforms.ToTensor(),
     # transform.normalize (mean for all channels) (std for all channels)
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # (image-mean)/std
)

# Data Generation
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
        img_name = (self.path + '/' + self.data.iloc[idx, 1])
        print(img_name)
        img = cv2.imread(img_name)
        img = img.transpose((2, 0, 1))
        print("Length ", len(img))
      
        labels = self.data.iloc[idx, 2]
       
        labels = np.array([labels])
        labels = labels.reshape(-1, 1)
        print("Labels ", labels, type(labels))
        sample = {'image': img, 'labels': labels}
        return sample

# Dataset Information
## The dataset contains 96 image samples. where 50% is original image and other 50% is forged version of those image.
## The CSV file contains image name and label (0/1) where 1 -> Forged and 0 -> Original
## All the images are resized to (1280 * 1240)

dir_train_img = "Image_Forged/benchmark_data/SelectedImages"
train_dataset = ForgedDataset(
    csv_file='C:/Users/BS304/Documents/CodeFiles/Forged_Image_Detection/classifications.csv', path=dir_train_img, transform=transforms, labels=True)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=0)

# The Model
class AlexNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input_ch = 3, output_ch = 64, Kernel_size = 5, stride = 4, padding = 3
        self.conv1 = nn.Conv2d(3, 512, 11, stride = 4, padding = 2)
        self.conv2 = nn.Conv2d(512, 256, 5, padding = 2)
        self.conv3 = nn.Conv2d(256, 128, 5, padding = 2)
        self.conv4 = nn.Conv2d(128, 64, 3, padding = 1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding = 1)
        
        self.fc1 = nn.Linear(32 * 6 * 6, 256) # 16 * 5 * 5
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.pool = nn.MaxPool2d(kernel_size= 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace = True))
         #print("Conv1: ", x.shape)
        x = self.pool(F.relu(self.conv2(x), inplace = True))
        #print("Conv2: ", x.shape)
        x = F.relu(self.conv3(x), inplace = True)
         #print("Conv3: ", x.shape)
        x = F.relu(self.conv4(x), inplace = True)
         #print("Conv4: ", x.shape)
        x = self.pool(F.relu(self.conv5(x), inplace = True))
        x = self.avgpool(x)
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = AlexNet()
net.to(device)

# Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# Model Training
print(len(data_loader_train))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for epoch in range(10):
    
    print('Epoch {}/{}'.format(epoch, 3 - 1))
    print('-' * 10)

    net.train()    
    tr_loss = 0
    

    for step, batch in enumerate(data_loader_train):

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
       
        outputs = net(inputs)
        print("Shape : " , len(outputs), len(labels))
        loss = criterion(outputs, labels.squeeze())


        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        
        if epoch == 1 and step > 19:
            epoch_loss = tr_loss / 20
            print('Training Loss: {:.4f}'.format(epoch_loss))
            break

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))

### Loss for this model after 10 Epochs => 0.69 
### It's obvious that the loss is so high as we do not have enough data to train as well as no significant image processing