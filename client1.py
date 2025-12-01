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
import copy
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
lr = 0.003
batch_size = 32
host_ip = "127.0.0.1"
port = 8081

# 강한 데이터 증강 (Label 1 데이터 활용 극대화)
train_transform = v2.Compose([
    v2.Resize(224, antialias=True),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
class MobileNetTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width_mult=0.35):
        super().__init__()
        # 기본 MobileNetV3-small 불러오기
        base = models.mobilenet_v3_small(weights=None)

        # width multiplier 적용
        def wm(ch): return max(int(ch * width_mult), 1)

        # 첫 Conv 레이어 축소
        base.features[0][0].out_channels = wm(16)

        # 중간 레이어 채널 축소
        for block in base.features:
            if hasattr(block, "block"):
                # expand, out 둘 다 줄임
                if hasattr(block.block[0], "in_channels"):
                    block.block[0].in_channels = wm(block.block[0].in_channels)
                if hasattr(block.block[-1], "out_channels"):
                    block.block[-1].out_channels = wm(block.block[-1].out_channels)

        # 마지막 단계 축소
        last_channels = wm(576)
        base.classifier[0] = nn.Linear(last_channels, wm(128))
        base.classifier[3] = nn.Linear(wm(128), num_classes)

        self.model = base

    def forward(self, x):
        return self.model(x)



def train(model, criterion, optimizer, train_loader):
    model.to(device)
    model.train()

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0
        
        # 클래스별 통계
        class_loss = [0.0] * NUM_CLASSES
        class_count = [0] * NUM_CLASSES

        for (images, labels) in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            # 클래스별 통계 수집
            for i in range(len(labels)):
                label_idx = labels[i].item()
                class_count[label_idx] += 1
                class_loss[label_idx] += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.double() / total

        print(f"Epoch [{epoch + 1}/{local_epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}%")
        
        # Label 1 학습 상태 모니터링
        if class_count[1] > 0:
            print(f"  -> Label 1: {class_count[1]} samples, Avg Loss: {class_loss[1]/class_count[1]:.4f}")
    
    return model

##############################################################################################################################


####################################################### 수정 가능 ##############################################################

class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.items = blob["items"]
        self.is_train = is_train
        self.transform = transform

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
    num_workers = min(4, os.cpu_count() or 4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        prefetch_factor=2, 
        persistent_workers=True,
        drop_last=True
    )

    model = MobileNetTiny().to(device)

    # Label 1에 훨씬 더 높은 가중치 부여 (Client1이 Label 1을 완전히 책임짐)
    class_weights = torch.tensor([1.0, 3.5, 1.0, 1.0]).to(device)
    
    global_round = 20
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_round)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
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