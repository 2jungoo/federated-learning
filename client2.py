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
DATASET_NAME = "./dataset/client2.pt"
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 3
lr = 0.003
batch_size = 16
host_ip = "127.0.0.1"
port = 8081
mu = 0.001

################# 전처리 코드 #################
train_transform = v2.Compose([
    v2.Resize(224),
    v2.RandomCrop(IMG_SIZE),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(7),
    v2.ColorJitter(brightness=0.15, contrast=0.15),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])



class Network1(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        
        self.backbone = models.mobilenet_v2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes)
        )

       
        nn.init.xavier_normal_(self.backbone.classifier[1].weight)

    def forward(self, x):
        return self.backbone(x)


def train(model, criterion, optimizer, train_loader):   

    model.to(device)

   
    global_params = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for (images, labels) in tqdm(train_loader, desc=f"Train (FedProx, epoch {epoch+1})"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            
            loss = criterion(outputs, labels)

            prox_reg = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    
                    prox_reg = prox_reg + ((param - global_params[name]) ** 2).sum()

            loss = loss + (mu / 2.0) * prox_reg

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (running_corrects.double() / total) * 100.0

        print(f"[FedProx] Epoch [{epoch + 1}/{local_epochs}] "
              f"=> Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")

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
        x = rec["tensor"].float() / 255.0    
        y = int(rec["label"])

        x = self.transform(x)

        return x, y


def main():

    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=train_transform)
    num_workers = max(2, (os.cpu_count() or 8) - 2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    
        pin_memory=True,
        
    )

    model = Network1().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
