from PIL import Image
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from sklearn.model_selection import train_test_split
import torch


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


def process_data():
    pytorch_transforms = Compose(
        [Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    normal_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Normal", label=0, transform=pytorch_transforms)
    tb_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Pneumonia", label=2, transform=pytorch_transforms)

    normal_splits = random_split(normal_train_dataset, [3000, 2500, 2000, 1688])
    tb_splits = random_split(tb_train_dataset, [500, 400, 600, 288])
    pneumonia_splits = random_split(pneumonia_train_dataset, [1000, 1200, 900, 1045])

    partitions = [
    ConcatDataset([normal_splits[i], tb_splits[i], pneumonia_splits[i]]) for i in range(4)
]

    normal_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Normal", label=0, transform=pytorch_transforms)
    tb_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Pneumonia", label=2, transform=pytorch_transforms)
    test_dataset = ConcatDataset([normal_test_dataset, tb_test_dataset, pneumonia_test_dataset])
    
    return partitions, test_dataset


partitions, test_dataset = process_data()

for i, _ in enumerate(partitions):
    partition_train_data = partitions[i]

    # Split into training and validation
    train_size = int(0.8 * len(partition_train_data))
    val_size = len(partition_train_data) - train_size
    partition_train, partition_val = random_split(partition_train_data, [train_size, val_size])

    # Save the dataset as a pickle file
    train_dataset_name = "train" + str(i)+"dataset.pt"
    val_dataset_name = "val" + str(i)+"dataset.pt"
    torch.save(partition_train, train_dataset_name)
    torch.save(partition_val, val_dataset_name)
torch.save(test_dataset, "testset.pt")


def load_data(partition_id: int, batch_size: int):
    train_name = "train" + str(partition_id)+"dataset.pt"
    val_name = "val" + str(partition_id)+"dataset.pt"
  
    partition_train = torch.load(train_name)
    partition_val = torch.load(val_name)
    test_dataset = torch.load("testset.pt")

    # Create DataLoaders for train, validation
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(partition_val, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloader, valloader, testloader