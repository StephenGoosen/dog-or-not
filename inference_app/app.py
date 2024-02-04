import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys

num_classes = 1
image_size = (224, 224)
pretrained_weights = "IMAGENET1K_V2" 

class Inference:
    image_name = "doggie.jpg" 
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.get_device_name(0)
    device = torch.device("cuda")
    print("Currently Using GPU")
else:
    device = torch.device("cpu")
    print("Currently Using CPU")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
])

def Inference(model, image_path):
    model.eval()
    with torch.no_grad():
        input_image = Image.open(image_path)
        input_tensor = transform(input_image).unsqueeze(0).to(device)
        pred = model(input_tensor)
        prediction = torch.sigmoid(pred).item()

    if 0 <= prediction < 0.5:
        return "The image does not contain a dog"
    elif 0.5 <= prediction <= 1:
        return "The image contains a dog"
    else:
        return "Error: prediction out of bounds."

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 1280)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
class MobileNetV2_Model(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_Model, self).__init__()

        self.model_base = models.mobilenet_v2(weights=pretrained_weights)

        for params in self.model_base.parameters():
            params.requires_grad = False

        self.classifier = Classifier(1000, num_classes) 

    def forward(self, x):
        x = self.model_base(x)
        x = self.classifier(x)
        return x
    
def model_get(num_classes, device):
    model = MobileNetV2_Model(num_classes)
    model.to(device)
    return model

class DogClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog Classifier App")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.image_path = None

        self.label = tk.Label(root, text="Upload an image to check if it's a dog.")
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.classify_button = tk.Button(root, text="Classify", command=self.classify_image)
        self.classify_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

    def load_model(self):
        model_file = os.path.join(os.path.dirname(sys.argv[0]), 'model.pth')

        if os.path.exists(model_file):
            model_loc = "model.pth"
            model = model_get(num_classes, device=device).to(device)
            model.load_state_dict(torch.load(model_loc))

            return model
        else:
            print(f"Error: Model file '{model_file}' not found.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.show_image()

    def show_image(self):
        img = Image.open(self.image_path)
        img = img.resize((300, 300)) 
        img = ImageTk.PhotoImage(img)
        self.label.config(image=img)
        self.label.image = img

    def classify_image(self):
        if self.image_path:
            threading.Thread(target=self.inference_and_update_label, args=(self.image_path,)).start()
        else:
            self.result_label.config(text="Please upload an image first.")

    def inference_and_update_label(self, image_path):
        result = self.inference_function(image_path)
        self.result_label.config(text=result)
        
    def inference_function(self, image_path):
        return Inference(self.model, image_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = DogClassifierApp(root)
    root.mainloop()