"""flower-project: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from imbalanced.load_data import load_data, CustomDataset
from imbalanced.net import Net, get_weights, set_weights, test, train
#from torch.utils.data import Dataset
#import os
#from PIL import Image

"""
class CustomDataset(Dataset):
    def __init__(self, data_dir, label, transform=None):
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
        self.image_paths = [
            os.path.join(data_dir, fname) for fname in os.listdir(data_dir)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"img": image, "label": self.label}
"""

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, testloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        print("val report")
        loss, accuracy = test(self.net, self.valloader, self.device)
        print(f"print client: Val Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    print("loading data")
    trainloader, valloader, testloader = load_data(partition_id, num_partitions, batch_size)
    print("loaded data")
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, testloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
