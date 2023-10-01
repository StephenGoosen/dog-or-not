import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging

import utils
import config
import dataset
from dataset import CustomImageDataset as CID
from model import model_get

'''
Logs will be saved in logging
'''
logging.basicConfig(
    filename='training.log',  # Specify the log file name
    level=logging.INFO,       # Set the logging level to INFO or DEBUG
    format='%(asctime)s [%(levelname)s]: %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'
)

'''
Set to CUDA if available, otherwise use CPU
'''
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.get_device_name(0)
    device = torch.device("cuda")  # Set the device to GPU
else:
    device = torch.device("cpu")  # Set the device to CPU

'''
The hyperparameters are controlled by config.py. 
'''
num_epochs = config.Train.num_epochs
batch_size = config.Train.batch_size
learning_rate = config.Train.learning_rate
num_classes = config.Model.num_classes
train_transform = dataset.train_transform
val_transform = dataset.val_transform
accuracy_fn = utils.accuracy_fn
'''
Class weights are calculated and then passed into the training function's loss_fn()
'''

train_dataset = CID(root_dir='data/train', transform=train_transform)
val_dataset = CID(root_dir='data/validation', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

total = len(train_dataset.labels)
len0 = train_dataset.labels.count(0)
len1 = train_dataset.labels.count(1)
weight_for_0 = (1 / len0) * (total / 2.0)
weight_for_1 = (1 / len1) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
scale_factor = 1.0 / class_weight[0]
scaled_class_weights = class_weight[1] * scale_factor

model = model_get(num_classes=num_classes, device=device)
loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor(scaled_class_weights)).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

'''
Initialization of train/val loss and accuracy lists
'''
train_losses = [] 
train_accuracies = []
val_losses = []
val_accuracies= []

epochs = config.Train.num_epochs

for epoch in tqdm(range(epochs)):
    logging.info(f"Epoch: {epoch}\n----------")

    train_loss, train_acc = utils.train_step(model=model,
                                             data_loader=train_loader,
                                             loss_fn=loss_fn,
                                             optimizer=optimizer,
                                             accuracy_fn=accuracy_fn,
                                             device=device
    )
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_loss, val_acc = utils.val_step(model=model,
                                       data_loader=val_loader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device=device
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

'''
Move train/val loss and accuracy measure to CPU to be plotted
'''
train_losses = [loss.cpu().item() for loss in train_losses]
val_losses = [loss.cpu().item() for loss in val_losses]
train_accs = [acc.cpu().item() for acc in train_accuracies]
val_accs = [acc.cpu().item() for acc in val_accuracies]

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('accuracy_plot.png')





