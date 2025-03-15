from PIL import Image
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
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
    print("processing data")
    pytorch_transforms = Compose(
        [Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    normal_train_dataset0 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part1/0", label=0, transform=pytorch_transforms)
    tb_train_dataset0 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part1/1", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset0 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part1/2", label=2, transform=pytorch_transforms)
    partition0 = ConcatDataset([normal_train_dataset0, tb_train_dataset0, pneumonia_train_dataset0])
    
    normal_train_dataset1 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part2/0", label=0, transform=pytorch_transforms)
    tb_train_dataset1 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part2/1", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset1 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part2/2", label=2, transform=pytorch_transforms)
    partition1 = ConcatDataset([normal_train_dataset1, tb_train_dataset1, pneumonia_train_dataset1])

    normal_train_dataset2 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part3/0", label=0, transform=pytorch_transforms)
    tb_train_dataset2 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part3/1", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset2 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part3/2", label=2, transform=pytorch_transforms)
    partition2 = ConcatDataset([normal_train_dataset2, tb_train_dataset2, pneumonia_train_dataset2])

    normal_train_dataset3 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part4/0", label=0, transform=pytorch_transforms)
    tb_train_dataset3 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part4/1", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset3 = CustomDataset("C:/Users/14871/Downloads/data/split_data/Part4/2", label=2, transform=pytorch_transforms)
    partition3 = ConcatDataset([normal_train_dataset3, tb_train_dataset3, pneumonia_train_dataset3])

    normal_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Normal", label=0, transform=pytorch_transforms)
    tb_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Pneumonia", label=2, transform=pytorch_transforms)
    test_dataset = ConcatDataset([normal_test_dataset, tb_test_dataset, pneumonia_test_dataset])
    print("processed data")
    
    return partition0, partition1, partition2, partition3, test_dataset


partition0, partition1, partition2, partition3, test_dataset = process_data()

train_size0 = int(0.8 * len(partition0))
val_size0 = len(partition0) - train_size0
partition_train0, partition_val0 = random_split(partition0, [train_size0, val_size0])

train_size1 = int(0.8 * len(partition1))
val_size1 = len(partition1) - train_size1
partition_train1, partition_val1 = random_split(partition1, [train_size1, val_size1])

train_size2 = int(0.8 * len(partition2))
val_size2 = len(partition2) - train_size2
partition_train2, partition_val2 = random_split(partition2, [train_size2, val_size2])

train_size3 = int(0.8 * len(partition3))
val_size3 = len(partition3) - train_size3
partition_train3, partition_val3 = random_split(partition3, [train_size3, val_size3])


def load_data(partition_id: int, batch_size: int):
    print("loading data")
    if partition_id == 0:
        partition_train = partition_train0
        partition_val = partition_val0
    
    if partition_id == 1:
        partition_train = partition_train1
        partition_val = partition_val1

    if partition_id == 2:
        partition_train = partition_train2
        partition_val = partition_val2

    if partition_id == 3:
        partition_train = partition_train3
        partition_val = partition_val3

    # Create DataLoaders for train, validation
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(partition_val, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    print("loaded data")

    return trainloader, valloader, testloader