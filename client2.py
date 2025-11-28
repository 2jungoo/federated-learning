import socket
import pickle
from tqdm import tqdm
import time
import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import struct
from collections import OrderedDict
import warnings
import select
import os
from torchvision import models
import torchvision.transforms.v2 as v2
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client2.pt"
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 2
lr = 0.002
batch_size = 128
host_ip = "127.0.0.1"
port = 8081

################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
train_transform = v2.Compose([
    v2.Resize((64, 64), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Network1(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def train(model, criterion, optimizer, train_loader):
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scaler = GradScaler()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
                                              epochs=local_epochs)

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for (images, labels) in tqdm(train_loader, desc=f"Client2 Ep {epoch + 1}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient Masking (Classifier[2] is Linear)
            if model.classifier[2].weight.grad is not None:
                model.classifier[2].weight.grad[1, :] = 0.0
            if model.classifier[2].bias.grad is not None:
                model.classifier[2].bias.grad[1] = 0.0

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    return model


##############################################################################################################################


####################################################### 수정 가능 ##############################################################


class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        print(f"Loading & Caching {pt_path}...")
        blob = torch.load(pt_path, map_location="cpu")
        items = blob["items"]

        self.data = []

        pre_process = v2.Compose([
            v2.Resize((64, 64), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for item in tqdm(items, desc="Pre-processing"):
            img = item["tensor"].float() / 255.0
            img = pre_process(img)
            label = int(item["label"])
            self.data.append((img, label))

        self.augment = v2.RandomHorizontalFlip(p=0.5) if is_train else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx]
        if self.augment:
            img = self.augment(img)
        return img, label


def main():
    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=train_transform)
    num_workers = 0

    # [수정] prefetch_factor 삭제
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Network1().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    ##############################################################################################################################

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

######################################################################################################################