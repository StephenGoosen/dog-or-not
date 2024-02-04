class Train:
    num_epochs = 2
    batch_size = 64
    learning_rate = 3e-4

class Model:
    num_classes = 1 #Set to 1, because a BCE and sigmoid function are used.
    pretrained_weights = "IMAGENET1K_V2" #Set pretrained weights
    trainable = False  #Setting to true, will increase training time.

class Data:
    train_data_dir = 'data/train' #Training image location. Should contain 2 folders, 1 for each class.
    val_data_dir = 'data/validation' #Validation image location. Should contain 2 folders, 1 for each class.
    image_size = (224, 224) 

class Inference:
    image_name = "doggie.jpg" #Change name of image you want to test. Default image location in Main folder.