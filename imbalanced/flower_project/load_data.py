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


def load_data(partition_id: int, num_partitions: int):
    pytorch_transforms = Compose(
        [Resize((32, 32)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    normal_dataset = CustomDataset("data/Normal", label=0, transform=pytorch_transforms)
    tb_dataset = CustomDataset("data/Tuberculosis", label=1, transform=pytorch_transforms)
    full_dataset = ConcatDataset([normal_dataset, tb_dataset])
    partition_size = len(full_dataset) // num_partitions
    partitions = random_split(full_dataset, [partition_size] * num_partitions)
    partition_train_test = random_split(partitions[partition_id], [int(0.8 * partition_size), int(0.2 * partition_size)])
    trainloader = DataLoader(partition_train_test[0], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test[1], batch_size=32)
    return trainloader, testloader