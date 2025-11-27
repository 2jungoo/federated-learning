import socket
import pickle
from tqdm import tqdm
import time
import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
import struct
from collections import OrderedDict
import warnings
import select
import os
from torchvision import models
import torchvision.transforms.v2 as v2
import numpy as np
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client1.pt"
######################################################################################################

############################################# 수정 가능 #############################################
local_epochs = 5  # Client 1은 많이 학습
lr = 0.005
batch_size = 128
host_ip = "127.0.0.1"
port = 8081

train_transform = v2.Compose([
    v2.Resize((64, 64), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# [최종 모델] Depthwise Separable Convolution 적용
class EfficientFederatedNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientFederatedNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True)
        )

        # Block 1: 32 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True)
        )

        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True)
        )

        # Block 3: 128 -> 256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU6(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def train(model, criterion, optimizer, train_loader):
    model.to(device)
    model.train()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
                                              epochs=local_epochs)

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for (images, labels) in tqdm(train_loader, desc=f"Client1 Epoch {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_acc = running_corrects.double() / total
        print(f"C1 [{epoch + 1}] Loss: {running_loss / len(train_loader):.3f} Acc: {epoch_acc * 100:.2f}%")

    return model


##############################################################################################################################

class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        blob = torch.load(pt_path, map_location="cpu")
        self.items = blob["items"]
        self.is_train = is_train
        self.transform = transform
        self.labels = [int(item["label"]) for item in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        x = rec["tensor"].float() / 255.0
        y = int(rec["label"])
        x = self.transform(x)
        return x, y


def main():
    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=train_transform)
    num_workers = min(4, os.cpu_count())

    # [중요] Sampler: 데이터 불균형 완벽 해결
    class_counts = [422, 1339, 489, 606]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    model = EfficientFederatedNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    ########################################################### 수정 금지 2 ##############################################################
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host_ip, port))

    while True:
        data_size = struct.unpack('>I', client.recv(4))[0]
        rec_payload = b""

        remaining_payload = data_size
        while remaining_payload != 0:
            rec_payload += client.recv(remaining_payload)
            remaining_payload = data_size - len(rec_payload)
        dict_weight = pickle.loads(rec_payload)
        weight = OrderedDict(dict_weight)
        print("\nReceived updated global model from server")

        model.load_state_dict(weight, strict=True)

        read_sockets, _, _ = select.select([client], [], [], 0)
        if read_sockets:
            print("Federated Learning finished")
            break

        model = train(model, criterion, optimizer, train_loader)

        model_data = pickle.dumps(dict(model.state_dict().items()))
        client.sendall(struct.pack('>I', len(model_data)))
        client.sendall(model_data)

        print("Sent updated local model to server.")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nThe model will be running on", device, "device")
    time.sleep(1)
    main()