"""flower-project: A Flower / PyTorch app."""

from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, resnet18
from torchvision.models import ResNet18_Weights
from imbalanced.loss import RatioLoss


class Net(nn.Module):
    """Transfer Learning Network using Pretrained ResNet18"""

    def __init__(self):
        super(Net, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the fully connected layer to match the number of classes (3)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 3)  # 3 classes: Normal, TB, Pneumonia

        # Freeze the early layers (conv1, layer1, layer2, layer3)
        for param in self.resnet.conv1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)  # Returns raw logits (no softmax)
    

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
    criterion = RatioLoss(class_num=3, alpha=class_weights)
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
    criterion = torch.nn.CrossEntropyLoss()
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
    print(report)
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def global_test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    #criterion = torch.nn.CrossEntropyLoss().to(device) # add to device
    criterion = torch.nn.CrossEntropyLoss()
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
    print(report)
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy