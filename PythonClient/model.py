# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:51:56 2021

@author: akpo2
"""

# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

# for reading and displaying images
import matplotlib.pyplot as plt
from matplotlib import style

from numpy import genfromtxt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
import cv2
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import *
from torch.utils.data import Dataset, DataLoader


print("Libraries imported - ready to use PyTorch", torch.__version__)


training_folder_name = r"D:\carla\CARLA_0.8.4\train_data"
image_folder = os.path.join(training_folder_name,r"CameraRGBCopy")
csv_file = os.path.join(training_folder_name,"data_train.csv")
#for file in os.listdir(image_folder):
#    print(file)

# All images are 128x128 pixels
img_size = (128,128)

class Load_Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.physical_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.transform_csv = transforms.Compose([transforms.ToTensor()])
        
#    def transform(self, image, mask):
#        # Resize
#        # Transform to tensor
#        image = TF.to_tensor(image)
#        mask = TF.to_tensor(mask)
#        return image, mask

    def __len__(self):
        return len(self.physical_data)

    def __getitem__(self, idx):
        data = self.physical_data.iloc[idx,:]
        frame = (data["frame_count"])
#        print(frame)
        if frame <= 9534:
            frame = str(int(frame))
        else:
            frame = str(int(frame))+"_processed"
        img_path = os.path.join(self.root_dir,
                                frame+".png")
#        plt.imshow(img_name)
#        print(img_path)
#        image = io.imread(img_name)
#        image_resized = resize(image, (128,128),
#                       anti_aliasing=True)
#        image = image_resized #.astype(np.uint8)
#        print(image.shape)
#        plt.imshow(image)
#        PIL_image = Image.fromarray(image)
        
        image = Image.open(open(img_path, 'rb'))
        new_size = (128,128)
        image = image.resize(new_size)
        steer = self.physical_data.iloc[idx, 8]
#        print(steer, type(steer))
        steer = torch.tensor(np.array([steer]))
#        print(type(steer))
#        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'steer': steer}
#        print(sample['image'])
#        print("applying transforms ..............................................")
        if self.transform:
            sample_image = self.transform(sample['image'])
            sample['image'] = sample_image
#            sample['steer'] = self.transform_csv(sample['steer'])
        return ((sample_image,steer))
    

def transform_dataset(full_dataset):
    # Load all the images
    

    # Load all of the images, transforming them
#    full_dataset = torchvision.datasets.ImageFolder(
##        root=data_path,
#        transform=transformation
#    )
    
    
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=0,
        shuffle=True
    )
    
    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=True
    )
        
    return train_loader, test_loader




#####################################################################################################

#for file in os.listdir(image_folder):
#    path = os.path.join(image_folder,file)
#    image = cv2.imread(path)
#    


# Recall that we have resized the images and saved them into
transformation = transforms.Compose([
#        transforms.ToPILImage(),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
data = Load_Data(csv_file=csv_file,root_dir=image_folder,transform=transforms.ToTensor())

#for i in range(len(data)):
#    sample = data[i]
#	
#    print(i, sample['image'].shape, sample['landmarks'].shape)
#
#    ax = plt.subplot(1, 4, i + 1)
#    plt.tight_layout()
#    ax.set_title('Sample #{}'.format(i))
#    ax.axis('off')
#    if i == 3:
#        plt.show()
#        break
# Get the iterative dataloaders for test and training data
train_loader, test_loader = transform_dataset(data)
batch_size = train_loader.batch_size
print("Data loaders ready to read", training_folder_name)

####################################################################################################
# Create a neural net class
#class Net(nn.Module):
#    
#    
#    # Defining the Constructor
#    def __init__(self, num_classes=1):
#        super(Net, self).__init__()
#        
#        # In the init function, we define each layer we will use in our model
#        
#        # Our images are RGB, so we have input channels = 3. 
#        # We will apply 12 filters in the first convolutional layer
#        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=1)
#        
#        # A second convolutional layer takes 12 input channels, and generates 24 outputs
#        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=1)
#        
#        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5, padding=1)
#        # We in the end apply max pooling with a kernel size of 2
#        self.pool = nn.MaxPool2d(kernel_size=(3,3))
#        
#        # A drop layer deletes 20% of the features to help prevent overfitting
#        self.drop = nn.Dropout2d(p=0.5)
#        
#        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
#        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them
#        
#        # We need to flatten these in order to feed them to a fully-connected layer
#        self.fc1 = nn.Linear(in_features=110592, out_features=512)
#        self.fc2 = nn.Linear(in_features=512, out_features=256)
#        self.fc3 = nn.Linear(in_features=256, out_features=128)
#        self.fc4 = nn.Linear(in_features=128, out_features=1)
#
#    def forward(self, x):
#        # In the forward function, pass the data through the layers we defined in the init function
#        
#        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
#        x = F.elu(self.pool(self.conv1(x))) 
#        
#        # Use a ReLU activation function after layer 2
#        x = F.elu(self.pool(self.conv2(x))) 
#        
#        x = F.elu(self.pool(self.conv3(x)))  
#        
#        # Select some features to drop to prevent overfitting (only drop during training)
#        x = F.dropout(self.drop(x), training=self.training)
#        
#        # Flatten
##        x = x.view(-1, 32 * 32 * 24)
#        x = torch.flatten(x)
#        print(x.shape)
#        # Feed to fully-connected layer to predict class
#        x = self.fc1(x)
#        
#        x = self.fc2(x)
#        
#        x = self.fc3(x)
#        
#        x = self.fc4(x)
#        # Return class probabilities via a log_softmax function 
##        return torch.log_softmax(x, dim=1)
#        return x


class DriverNet(nn.Module):

  def __init__(self):
        super(DriverNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=5184, out_features=100),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=100, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )
        

  def forward(self, input):
      input = input.view(input.size(0), 3, 128, 128)
      output = self.conv_layers(input)
      output = output.view(output.size(0), -1)
#      print("output shape",output.shape)
      output = self.linear_layers(output)
      return output

    
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# Create an instance of the model class and allocate it to the device
model = DriverNet().to(device)

print(model)


####################################################################################################
def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader,0):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
#        print("in training...........")
#        print(type(data))
#        print(type(target))
        data, target = Variable(data.to(device)), Variable(target.to(device))
#        print("data shape",data.shape)
#        print("target shape",target.shape)
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
#        print("output shape",output.shape)
#        print("target shape",target.shape)
        
        output = output.type(torch.DoubleTensor)
        target = target.type(torch.DoubleTensor)
        # Get the loss
        loss = loss_criteria(output, target)
        

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss

# Use an "Adam" optimizer to adjust weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Specify the loss criteria
#loss_criteria = nn.CrossEntropyLoss()
loss_criteria = loss = nn.MSELoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 10 epochs (We restrict to 10 for time issues)
save_path = os.path.join(training_folder_name,"model")
epochs = 100
print('Training on', device)
best_loss = 10000.0
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
         # If the validation loss is at a minimum
        if test_loss < best_loss:
          # Save the model
          torch.save(model.state_dict(), os.path.join(save_path,"best.pth"))
          best_loss = test_loss
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
#creating dataframe and record all the losses and accuracies at each epoch
log_frame = pd.DataFrame(columns = ["Epoch", "Train Loss", "Test Loss"])
log_frame["Epoch"] = epoch_nums
log_frame["Train Loss"] = training_loss
log_frame["Test Loss"] = validation_loss
log_frame.to_csv(os.path.join(training_folder_name, "log2.csv"), index = False)



data = genfromtxt(os.path.join(training_folder_name, "log2.csv"),delimiter=',', names=['Epoch', 'Train Loss', 'Test Loss'])
epoch_list = []
train_loss_list = []
test_loss_list = []
for row in data:
  if not np.isnan(row[0]):
    epoch_list.append(row[0])
    train_loss_list.append(row[1])
    test_loss_list.append(row[2])
  

plt.plot(epoch_list, train_loss_list, label = "Training Loss")
plt.plot(epoch_list, test_loss_list, label = "Testing Loss")

plt.title('MSE Loss Vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.show()