import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import pytesseract


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################################################
# 1 Data Loading

def get_relevant_indices(dataset,classes, target_classes):
  indices = []
  for i in range(len(dataset)):
        # Check if the label is in the target classes
        label_index = dataset[i][1] # ex: 3
        label_class = classes[label_index] # ex: 'cat'
        if label_class in target_classes:
            indices.append(i)
  return indices


def get_data_loader(target_classes, batch_size):
    classes = ('LeftSide','RightSide')
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [ 
        transforms.Resize((1024, 734)),   
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load the training data
    trainset = torchvision.datasets.ImageFolder(root="pokemon_cards_2011_2023", transform=transform)
    print(trainset.class_to_idx)
    # Get the list of indices to sample from
    relevant_indices = get_relevant_indices(trainset, classes, target_classes)

    # Split into train, validation AND testing
    np.random.seed(1000)
    np.random.shuffle(relevant_indices)
    n = len(relevant_indices)

    train_end = max(1, int(0.7 * n))    # at least 1 sample
    val_end = max(train_end + 1, int(0.85 * n))  # at least 1 sample in val

    relevant_train_indices = relevant_indices[:train_end]
    relevant_val_indices = relevant_indices[train_end:val_end]
    relevant_test_indices = relevant_indices[val_end:]

    print("Total relevant images:", len(relevant_indices))
    print("Train:", len(relevant_train_indices), "Val:", len(relevant_val_indices), "Test:", len(relevant_test_indices))

    # In case test set is empty (small dataset)
    if len(relevant_test_indices) == 0 and val_end < n:
        relevant_test_indices = relevant_indices[val_end:n]

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=SubsetRandomSampler(relevant_train_indices), num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=SubsetRandomSampler(relevant_val_indices), num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=SubsetRandomSampler(relevant_test_indices), num_workers=1)

    return train_loader, val_loader, test_loader, classes




###############################################################################
# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path


def evaluate(net, data_loader, criterion, device):
    net.eval()
    total_loss = 0.0
    total_err = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            total_err += (preds != labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    err = total_err / total_samples if total_samples > 0 else 0.0
    loss = total_loss / total_samples if total_samples > 0 else 0.0
    return err, loss

###############################################################################
# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()



    #################### VISUALIZING DATA
def visalizeData():
    train_loader, val_loader, test_loader, classes = get_data_loader(
        target_classes=['LeftSide','RightSide'],
        batch_size=1)  # One image per batch

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))

    k = 0
    for images, labels in train_loader:
        image = images[0]
        img = np.transpose(image.numpy(), [1, 2, 0])
        img = img / 2 + 0.5  # unnormalize

        plt.subplot(3, 5, k+1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(classes[labels.item()], fontsize=10)

        k += 1
        if k >= 15:  # show first 15 images
            break

    plt.tight_layout()
    plt.show()


class LargeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LargeNet, self).__init__()
        self.name = "large"

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                       # logits
        return x


def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    target_classes = ["RightSide","LeftSide"]
    torch.manual_seed(1000)

    # Load data
    train_loader, val_loader, test_loader, classes = get_data_loader(target_classes, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0.0
        total_train_err = 0.0
        total_samples = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total_train_err += (preds != labels).sum().item()
            total_train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        train_err[epoch] = total_train_err / total_samples
        train_loss[epoch] = total_train_loss / total_samples

        # Validation loop
        net.eval()
        val_total_loss = 0.0
        val_total_err = 0.0
        val_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_total_err += (preds != labels).sum().item()
                val_total_loss += loss.item() * labels.size(0)
                val_samples += labels.size(0)

        val_err[epoch] = val_total_err / val_samples
        val_loss[epoch] = val_total_loss / val_samples

        print(f"Epoch {epoch+1}: Train err: {train_err[epoch]:.4f}, Train loss: {train_loss[epoch]:.4f} | "
              f"Validation err: {val_err[epoch]:.4f}, Validation loss: {val_loss[epoch]:.4f}")

        # Save checkpoint
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)

    elapsed_time = time.time() - start_time
    print(f"Finished Training. Total time elapsed: {elapsed_time:.2f} seconds")

    # Save training stats
    np.savetxt(f"{model_path}_train_err.csv", train_err)
    np.savetxt(f"{model_path}_train_loss.csv", train_loss)
    np.savetxt(f"{model_path}_val_err.csv", val_err)
    np.savetxt(f"{model_path}_val_loss.csv", val_loss)

    return net, (train_err, train_loss, val_err, val_loss), device


################## CROPPING THE AREA WHERE THE ID IS 
def cropID(image, corner='left', crop_size=(50, 20)):
    _, H, W = image.shape
    cw, ch = crop_size
    if corner == 'left':
        x_start, y_start = 0, H - ch
    else:  # Cropping the image to the right
        x_start, y_start = W - cw, H - ch
    cropped = image[:, y_start:y_start+ch, x_start:x_start+cw]
    return cropped

def predictSideOfID(model_path,img_path):
    model = LargeNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    transform = transforms.Compose([
        transforms.Resize((1024, 734)),             
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(img_path).convert('RBG')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 0:
            side = "Left" 
        else: 
            side = "Right"
        print(f"Predicted side is {side}")
        return side 
    

# OCR That reads the ID after predicting the side
def readID(img_path, side):
    image = Image.open(img_path).convert("RGB")
    cropped = cropID(image, corner=side.lower())
    card_id = pytesseract.image_to_string(cropped, config ='--psm 7').strip()
    print(f"Detected card ID: {card_id}")
    return card_id
