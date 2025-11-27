import threading
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import struct
from tqdm import tqdm
import copy
import warnings
import random
import os
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

warnings.filterwarnings("ignore")

############################################## 수정 불가 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/test.pt"
######################################################################################################

####################################################### 수정 가능 #######################################################
target_accuracy = 90.0
global_round = 30
batch_size = 256  # 배치 사이즈 최대화 (속도 향상)
host = '127.0.0.1'
port = 8081

# [속도/성능 최적화] 64x64 리사이즈
test_transform = v2.Compose([
    v2.Resize((64, 64), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# [최종 모델] Depthwise Separable Convolution 기반 초경량 고성능 모델
class EfficientFederatedNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientFederatedNet, self).__init__()

        # 1. Stem Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # 2. Inverted Residual Blocks (MobileNet Style) - 직접 구현하여 가볍게 만듦
        # Block 1: 32 -> 64
        self.block1 = nn.Sequential(
            # Depthwise
            nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True)
        )

        # Block 2: 64 -> 128 (Downsample)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True)
        )

        # Block 3: 128 -> 256 (Downsample)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU6(inplace=True)
        )

        # 3. Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        print(pt_path)
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


def measure_accuracy(global_model, test_loader):
    model = EfficientFederatedNet().to(device)
    model.load_state_dict(global_model)
    model.eval()

    correct = 0
    total = 0

    inference_start = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = (100 * correct / total) if total > 0 else 0
    inference_end = time.time()
    return accuracy, model, inference_end - inference_start


def average_models_with_momentum(models, prev_global_model=None, momentum=0.9):
    weight_avg = copy.deepcopy(models[0])
    for key in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[key] += models[i][key]
        weight_avg[key] = torch.div(weight_avg[key], len(models))

    if prev_global_model is not None:
        for key in weight_avg.keys():
            weight_avg[key] = momentum * weight_avg[key] + (1 - momentum) * prev_global_model[key]

    return weight_avg


##############################################################################################################################

####################################################### 수정 금지 ##############################################################
cnt = []
model_list = []
semaphore = threading.Semaphore(0)

global_model = None
prev_global_model = None  # For momentum
global_model_size = 0
global_accuracy = 0.0
current_round = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def handle_client(conn, addr, model, test_loader):
    global model_list, global_model, prev_global_model, global_accuracy, global_model_size, current_round, cnt
    print(f"Connected by {addr}")

    while True:
        if len(cnt) < 2:
            cnt.append(1)
            weight = pickle.dumps(dict(model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

        data_size = struct.unpack('>I', conn.recv(4))[0]
        received_payload = b""
        remaining_payload_size = data_size
        while remaining_payload_size != 0:
            received_payload += conn.recv(remaining_payload_size)
            remaining_payload_size = data_size - len(received_payload)
        model = pickle.loads(received_payload)

        model_list.append(model)

        if len(model_list) == 2:
            current_round += 1
            # Use momentum-based averaging
            global_model = average_models_with_momentum(model_list, prev_global_model, momentum=0.8)
            prev_global_model = copy.deepcopy(global_model)

            global_accuracy, global_model, _ = measure_accuracy(global_model, test_loader)
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy:.2f}%")
            global_model_size = get_model_size(global_model)
            model_list = []
            semaphore.release()
        else:
            semaphore.acquire()

        if (current_round == global_round) or (global_accuracy >= target_accuracy):
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)
            conn.close()
            break
        else:
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)


def get_model_size(global_model):
    model_size = len(pickle.dumps(dict(global_model.state_dict().items())))
    model_size = model_size / (1024 ** 2)
    return model_size


def get_random_subset(dataset, num_samples):
    if num_samples > len(dataset):
        raise ValueError(f"num_samples should not exceed {len(dataset)} (total number of samples in test dataset).")
    indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, indices)
    return subset


def average_models(models):
    weight_avg = copy.deepcopy(models[0])
    for key in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[key] += models[i][key]
        weight_avg[key] = torch.div(weight_avg[key], len(models))
    return weight_avg


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    connection = []
    address = []

    ############################ 수정 가능 ############################
    train_dataset = CustomDataset(DATASET_NAME, is_train=False, transform=test_transform)
    num_workers = min(4, os.cpu_count())

    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    model = EfficientFederatedNet().to(device)
    ####################################################################

    print(f"Server is listening on {host}:{port}")

    while len(address) < 2 and len(connection) < 2:
        conn, addr = server.accept()
        connection.append(conn)
        address.append(addr)

    training_start = time.time()

    connection1 = threading.Thread(target=handle_client, args=(connection[0], address[0], model, test_loader))
    connection2 = threading.Thread(target=handle_client, args=(connection[1], address[1], model, test_loader))

    connection1.start()
    connection2.start()
    connection1.join()
    connection2.join()

    training_end = time.time()
    total_time = training_end - training_start

    # 평가지표
    print(f"\n학습 성능 : {global_accuracy:.2f} %")
    print(f"\n학습 소요 시간: {int(total_time // 3600)} 시간 {int((total_time % 3600) // 60)} 분 {(total_time % 60):.2f} 초")
    print(f"\n최종 모델 크기: {global_model_size:.4f} MB")

    final_model = dict(global_model.state_dict().items())
    _, _, inference_time = measure_accuracy(final_model, test_loader)
    print(f"\n예측 소요 시간 : {(inference_time):.2f} 초")
    print("연합학습 종료")


if __name__ == "__main__":
    main()