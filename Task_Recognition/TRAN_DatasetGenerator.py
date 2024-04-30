import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import argparse

class TRAN_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def main(args):
    # Load the data
    data = pd.read_csv(args.data_path)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the TRAN_Dataset object
    dataset = TRAN_Dataset(data, transform=transform)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Now you can use the train_loader and val_loader to train and validate your ViT and Transformer models

if __name__ == "__main__":
    parser = argparse.ArgumentParser('TRAN_DatasetGenerator script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the data loaders')
    args = parser.parse_args()
    main(args)