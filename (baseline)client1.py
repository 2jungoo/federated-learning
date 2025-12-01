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

warnings.filterwarnings("ignore")


############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client1.pt"
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 5
lr = 0.001
batch_size = 32
host_ip = "127.0.0.1"
port = 8081


################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
train_transform = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), antialias=True), # network에 들어갈 image shape은 꼭 192 x 192로
    v2.RandomRotation(degrees=7),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

## 아래는 예시 모델이며, 예시 모델 그대로 사용하여 제출하면 안됨
class Network1(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5, 1, 1)
        self.bn1   = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, 5, 1, 1)
        self.bn2   = nn.BatchNorm2d(12)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(12, 24, 5, 1, 1)
        self.bn4   = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(24, 24, 5, 1, 1)
        self.bn5   = nn.BatchNorm2d(24)
        self.apool = nn.AdaptiveAvgPool2d((10,10))
        self.fc1   = nn.Linear(24*10*10, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.apool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train(model, criterion, optimizer, train_loader):   # pruning or quantization 적용시 필요한 경우 수정

    best_accuracy = 0.0

    model.to(device)

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for (images, labels) in tqdm(train_loader, desc="Train"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.double() / total

        print(f"Epoch [{epoch + 1}/{local_epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}%")

    return model

##############################################################################################################################



####################################################### 수정 가능 ##############################################################


class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        blob = torch.load(pt_path, map_location="cpu")
        self.items = blob["items"]
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        x = rec["tensor"].float() / 255.0      # uint8 [C,H,W]
        y = int(rec["label"])

        x = self.transform(x)

        return x, y

def main():

    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=train_transform)
    num_workers = max(2, (os.cpu_count() or 8) - 2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True,
                                               prefetch_factor=4, persistent_workers=True)

    model = Network1().to(device)


    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
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










