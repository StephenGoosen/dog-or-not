import torch
from torchvision import transforms
from PIL import Image

from dataset import val_transform
from model import model_get
from utils import Inference

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.get_device_name(0)
    device = torch.device("cuda")  # Set the device to GPU
    print("Currently Using GPU")
else:
    device = torch.device("cpu")  # Set the device to CPU
    print("Currently Using CPU")

'''
Load model architecture, weights, and info
'''
model_loc = "model.pth"
model_info_loc = "model_info.pth"

model = model_get(num_classes=1, device=device).to(device)
model.load_state_dict(torch.load(model_loc))

'''
Load Image
'''
input_image = Image.open('puppy.jpg') #select input image
image = val_transform(input_image).unsqueeze(0).to(device) #expected extra dimension for batch

'''
Make Inference
'''

Inference(model, input_tensor=image)