# train.py

import torch
import torch.nn as nn

import utils
import config
import dataset
from model import model_get

'''
Set to CUDA if available, otherwise use CPU
'''

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.get_device_name(0)
    device = torch.device("cuda")  # Set the device to GPU
    print("Currently Using GPU")
else:
    device = torch.device("cpu")  # Set the device to CPU
    print("Currently Using CPU")

'''
The hyperparameters are controlled by config.py. 
'''
num_epochs = config.Train.num_epochs
batch_size = config.Train.batch_size
learning_rate = config.Train.learning_rate
num_classes = config.Model.num_classes

train_transform = dataset.train_transform
val_transform = dataset.val_transform
train_dataset = dataset.train_dataset
val_dataset = dataset.val_dataset
train_loader = dataset.train_loader
val_loader = dataset.val_loader
scaled_class_weights = dataset.scaled_class_weights

accuracy_fn = utils.accuracy_fn

'''
Class weights are calculated and then passed into the training function's loss_fn()
'''

model = model_get(num_classes=num_classes, device=device) #Create model. Function is in model.py
loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor(scaled_class_weights)).to(device) #Use BCE with logits, load class weights
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate) #Adam optimizer w/o weight decay

'''
Initialization of train/val loss and accuracy lists
'''

train_losses = [] 
train_accuracies = []
val_losses = []
val_accuracies= []

epochs = config.Train.num_epochs #epochs set in config.py

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n----------")

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
Move train/val loss and accuracy measure to CPU to be plotted using PlotLossAcc function in utils
'''
train_losses = [loss.cpu().item() for loss in train_losses]
val_losses = [loss.cpu().item() for loss in val_losses]
train_accs = [acc.cpu().item() for acc in train_accuracies]
val_accs = [acc.cpu().item() for acc in val_accuracies]

utils.PlotLossAcc(train_losses, val_losses, train_accs, val_accs)

'''
Export model to main folder
'''

model_info = {'model_state_dict': model.state_dict(),
              'class_to_idx': train_dataset.class_to_idx,
              'idx_to_class': {v: k for k, v in train_dataset.class_to_idx.items()}
              }


torch.save(model.state_dict(), 'model.pth')

torch.save(model_info, 'model_info.pth')

