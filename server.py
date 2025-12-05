#수정 후 
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
batch_size = 128
host = '127.0.0.1'
port = 8081

# [중요] 192x192 입력을 받도록 유지
test_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# [최적화 모델] 192x192 입력을 효율적으로 처리하는 구조 (약 0.4MB)
class Network1(nn.Module):
    def __init__(self, num_classes=4):
        super(Network1, self).__init__()

        self.features = nn.Sequential(
            # Input: 3 x 192 x 192
            # Stride 2를 사용하여 공간 해상도를 절반으로 즉시 축소 (속도 핵심)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # -> 16 x 96 x 96
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 16 x 48 x 48

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 32 x 24 x 24

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 64 x 12 x 12

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 128 x 6 x 6
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


# [속도 핵심] RAM Caching Dataset (로딩 바 1회 후 초고속 학습)
class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        print(f"Loading & Caching {pt_path} to RAM...")
        blob = torch.load(pt_path, map_location="cpu")
        items = blob["items"]

        self.data = []

        pre_process = v2.Compose([
            v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for item in tqdm(items, desc="Pre-processing"):
            img = item["tensor"].float() / 255.0
            img = pre_process(img)
            label = int(item["label"])
            self.data.append((img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def measure_accuracy(global_model, test_loader):
    model = Network1().to(device)
    model.load_state_dict(global_model)
    model.eval()

    correct = 0
    total = 0
    # inference_start = time.time() # 원본 코드 변수명 유지를 위해 주석 처리하거나 아래서 계산

    start = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = (100 * correct / total) if total > 0 else 0
    end = time.time()
    return accuracy, model, end - start



##############################################################################################################################

####################################################### 수정 금지 ##############################################################
cnt = []
model_list = []
semaphore = threading.Semaphore(0)

global_model = None
global_model_size = 0
global_accuracy = 0.0
current_round = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def handle_client(conn, addr, model, test_loader):
    global model_list, global_model, global_accuracy, global_model_size, current_round, cnt
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
            global_model = average_models(model_list)
            global_accuracy, global_model, _ = measure_accuracy(global_model, test_loader)
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy}%")
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
    # [중요] Worker 0 설정 (RAM Caching 데이터 사용 시 필수)
    num_workers = 0

    # 원본 코드 구조 유지하되 Worker 0 적용
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    model = Network1().to(device)
    ####################################################################

    print(f"Server is listening on {host}:{port}")

    while len(address) < 2 and len(connection) < 2:
        conn, addr = server.accept()
        connection.append(conn)
        address.append(addr)

    training_start = time.time()

    connection1 = threading.Thread(target=handle_client, args=(connection[0], address[0], model, test_loader))
    connection2 = threading.Thread(target=handle_client, args=(connection[1], address[1], model, test_loader))

    connection1.start();
    connection2.start()
    connection1.join();
    connection2.join()

    training_end = time.time()
    total_time = training_end - training_start

    print(f"\n학습 성능 : {global_accuracy} %")
    print(f"\n학습 소요 시간: {int(total_time // 3600)} 시간 {int((total_time % 3600) // 60)} 분 {(total_time % 60):.2f} 초")
    print(f"\n최종 모델 크기: {global_model_size:.4f} MB")

    final_model = dict(global_model.state_dict().items())
    _, _, inference_time = measure_accuracy(final_model, test_loader)
    print(f"\n예측 소요 시간 : {(inference_time):.2f} 초")

    print("연합학습 종료")


if __name__ == "__main__":
    main()
