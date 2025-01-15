"""flower-project: A Flower / PyTorch app."""

from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, resnet18


class ResNet(nn.Module):
    """Pretrained resnet with fine-tuning"""

    def __init__(self):
        super(ResNet, self).__init__()
        # Load pretrained resnet model
        self.resnet = resnet18(weights="IMAGENET1K_V1")

        # Fine-tune all layers (optional: freeze earlier layers)
        for param in self.resnet.parameters():
            param.requires_grad = True  # Enable gradient computation for all layers

        # Replace the classifier with a custom output layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 3) # 3 output layer for 3 classes

    def forward(self, x):
        # Pass input through resnet
        return self.resnet(x)


class Net(nn.Module):
    """Model (simple CNN)"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3) #output 3 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class DenseNet(nn.Module):
    """Pretrained DenseNet with fine-tuning"""

    def __init__(self):
        super(DenseNet, self).__init__()
        # Load pretrained DenseNet model
        self.densenet = densenet121(weights="IMAGENET1K_V1")

        # Fine-tune all layers (optional: freeze earlier layers)
        for param in self.densenet.parameters():
            param.requires_grad = False  # Enable gradient computation for all layers
        for param in self.densenet.features[-10:].parameters():
            param.requires_grad = True

        # Replace the classifier with a custom output layer
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 3) # 3 output layer for 3 classes

    def forward(self, x):
        # Pass input through DenseNet
        return self.densenet(x)
    

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    labels = [sample["label"] for sample in trainloader.dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0 # added
        for i, batch in enumerate(trainloader):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # added
        print(f"print net.train: Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}") # added print

    val_loss, val_acc = test(net, valloader, device)
    print(f"print net.val: Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device) # add to device
    all_labels, all_preds = [], []
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            preds = torch.max(outputs.data, 1)[1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Tuberculosis", "Pneumonia"], zero_division=0)
    print("test report" + report)
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
