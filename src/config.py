class Train:
    num_epochs = 3
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 0.0001

class Model:
    num_classes = 1
    pretrained_weights = "IMAGENET1K_V2"
    trainable = False

class Data:
    train_data_dir = 'data/train'
    val_data_dir = 'data/validation'
    image_size = (224, 224)