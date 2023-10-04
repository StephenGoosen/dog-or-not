from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os

import utils
import config

image_size = config.Data.image_size

'''
Setting up the data to be used by the torch.nn model should follow the structure below:

    --data
        --train
            --"Class 1"
            --"Class 2"
        --validation
            --"Class 1"
            --"Class 2"

As long as the class names are consistent, any two image classes can be used.
'''

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {"dog": 1, "not_dog": 0}
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, filename))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


'''
Random transforms are set up for the train set, but only the required transforms are done for the validation set.
All image inputs need to be tensors of the same size, 
and the normalization follows the ImageNet dataset mean and variance values.
'''

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
])

'''
These datasets set up the data to be loaded in batches of batch_size.
'''

train_dataset = CustomImageDataset(root_dir='data/train', transform=train_transform)
val_dataset = CustomImageDataset(root_dir='data/validation', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=config.Train.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.Train.batch_size, shuffle=False)

'''
The imbalance in the classes can affect quality of the model.
If the number of class 2 imagesoutweighed the number of class 1 images,
then classifying all images as 2 would achieve a high accuracy and would likely not converge.
scaled_class_weights is used in the loss function "weight" argument. 
'''

total = len(train_dataset.labels)
len0 = train_dataset.labels.count(0)
len1 = train_dataset.labels.count(1)
weight_for_0 = (1 / len0) * (total / 2.0)
weight_for_1 = (1 / len1) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
scale_factor = 1.0 / class_weight[0]
scaled_class_weights = class_weight[1] * scale_factor