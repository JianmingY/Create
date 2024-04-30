import torch
from torch import nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from TRAN_DatasetGenerator import TRAN_Dataset
from ViT_TRAN import ViT_Transformer


def train(model, train_loader, criterion, optimizer):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

    return loss.item()

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

    # Create the ViT_Transformer model
    model = ViT_Transformer(num_classes=args.num_classes, num_features=args.num_features).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train and evaluate the model
    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train_Vit_TRAN script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the data loaders')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--num_features', type=int, required=True, help='Number of features')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()
    main(args)