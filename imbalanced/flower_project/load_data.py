from PIL import Image
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


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


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    pytorch_transforms = Compose(
        [Resize((256, 256)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    normal_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Normal", label=0, transform=pytorch_transforms)
    tb_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Pneumonia", label=2, transform=pytorch_transforms)
    full_train_dataset = ConcatDataset([normal_train_dataset, tb_train_dataset, pneumonia_train_dataset])

    normal_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Normal", label=0, transform=pytorch_transforms)
    tb_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Pneumonia", label=2, transform=pytorch_transforms)
    test_dataset = ConcatDataset([normal_test_dataset, tb_test_dataset, pneumonia_test_dataset])

     # Split the training set into partitions for clients
    partition_size = len(full_train_dataset) // num_partitions
    partition_sizes = [partition_size] * num_partitions
    partition_sizes[-1] += len(full_train_dataset) % num_partitions  # Handle the remainder
    partitions = random_split(full_train_dataset, partition_sizes)
    
    # Retrieve the data partition for the given partition_id
    partition_train_data = partitions[partition_id]
    
    # Split into training and validation
    train_size = int(0.8 * len(partition_train_data))
    val_size = len(partition_train_data) - train_size
    partition_train, partition_val = random_split(partition_train_data, [train_size, val_size])
    
    # Create DataLoaders for train, validation, and test
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(partition_val, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, valloader, testloader



"""
removed validation

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    pytorch_transforms = Compose(
        [Resize((256, 256)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    normal_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Normal", label=0, transform=pytorch_transforms)
    tb_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_train_dataset = CustomDataset("C:/Users/14871/Downloads/data/train/Pneumonia", label=2, transform=pytorch_transforms)
    full_train_dataset = ConcatDataset([normal_train_dataset, tb_train_dataset, pneumonia_train_dataset])

    normal_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Normal", label=0, transform=pytorch_transforms)
    tb_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Tuberculosis", label=1, transform=pytorch_transforms)
    pneumonia_test_dataset = CustomDataset("C:/Users/14871/Downloads/data/test/Pneumonia", label=2, transform=pytorch_transforms)
    test_dataset = ConcatDataset([normal_test_dataset, tb_test_dataset, pneumonia_test_dataset])

    partition_size = len(full_train_dataset) // num_partitions
    partition_sizes = [partition_size] * num_partitions
    partition_sizes[-1] += len(full_train_dataset) % num_partitions  # Handle remainder
    partitions = random_split(full_train_dataset, partition_sizes)
    
    # Select the partition for the current client
    partition_train_data = partitions[partition_id]

    # Create DataLoaders for train, validation, and test
    trainloader = DataLoader(partition_train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader
"""