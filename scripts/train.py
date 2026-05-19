"""
Training script for PneumonMamba model.

Supports train/val/test splits with comprehensive evaluation metrics
including accuracy, precision, recall, and F1-score.
"""

import os
import random
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.pneumonmamba import VSSM


def seed_torch(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Train PneumonMamba')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classification classes')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='../checkpoints/best_model.pth', help='Model save path')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'small', 'base'],
                        help='Model variant: tiny, small, or base')
    return parser.parse_args()


def build_model(model_size, num_classes):
    """Build PneumonMamba model with specified size variant."""
    if model_size == 'tiny':
        model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
    elif model_size == 'small':
        model = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
    elif model_size == 'base':
        model = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=num_classes)
    return model


def main():
    args = get_args()
    seed_torch(seed=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Data augmentation and preprocessing
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "test": transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=data_transform["val"])
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'test'), transform=data_transform["test"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    test_num = len(test_dataset)

    # Save class-to-index mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(os.path.join(os.path.dirname(args.save_path), 'class_indices.json'), 'w') as json_file:
        json.dump(idx_to_class, json_file, indent=4)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    print(f"Using {train_num} images for training, {val_num} for validation, {test_num} for testing.")

    # Build model
    net = build_model(args.model_size, args.num_classes)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    train_steps = len(train_loader)
    torch.autograd.set_detect_anomaly(True)

    # Training loop
    for epoch in range(args.epochs):
        # Train phase
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss:.3f}"

        # Validation phase
        net.eval()
        acc = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                all_labels.extend(val_labels.numpy())
                all_preds.extend(predict_y.cpu().numpy())

        val_accurate = acc / val_num
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f} '
              f'val_accuracy: {val_accurate:.3f} '
              f'precision: {precision:.3f} '
              f'recall: {recall:.3f} '
              f'f1_score: {f1:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), args.save_path)

    print(f'Finished Training. Best validation accuracy: {best_acc:.3f}')

    # Test phase
    net.load_state_dict(torch.load(args.save_path))
    net.eval()
    test_acc = 0.0
    test_all_labels = []
    test_all_preds = []
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            test_acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            test_all_labels.extend(test_labels.numpy())
            test_all_preds.extend(predict_y.cpu().numpy())

    test_accurate = test_acc / test_num
    test_precision = precision_score(test_all_labels, test_all_preds, average='weighted')
    test_recall = recall_score(test_all_labels, test_all_preds, average='weighted')
    test_f1 = f1_score(test_all_labels, test_all_preds, average='weighted')

    print(f'Test Accuracy: {test_accurate:.3f} '
          f'Test Precision: {test_precision:.3f} '
          f'Test Recall: {test_recall:.3f} '
          f'Test F1 Score: {test_f1:.3f}')


if __name__ == '__main__':
    main()
