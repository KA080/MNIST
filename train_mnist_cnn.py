import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CNN 레이어를 정의하는 함수 (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
def SimpleCNNLayer(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),  # 3x3 커널
        nn.BatchNorm2d(out_features),  # 배치 정규화
        nn.ReLU(),  # 활성화 함수 ReLU
        nn.MaxPool2d(kernel_size=2)  # 2x2 풀링
    )

# 전체 CNN 분류기를 정의하는 함수
def SimpleCNNClassifier():
    return nn.Sequential(
        SimpleCNNLayer(1, 32),  # 입력 채널 1 (그레이스케일), 출력 채널 32
        SimpleCNNLayer(32, 32),  # 출력 채널 32 -> 32
        nn.Flatten(),  # 데이터를 평탄화하여 Fully Connected 레이어에 전달
        nn.Linear(1568, 64),  # Fully Connected Layer (입력 1568, 출력 64)
        nn.Linear(64, 10)  # 마지막 레이어, 10개의 클래스 (MNIST: 0~9 숫자)
    )

# 이미지를 텐서로 변환하는 전처리 과정 정의
transforms = transforms.ToTensor()

# MNIST 학습 데이터셋을 불러오고, 다운로드
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms
)

# MNIST 테스트 데이터셋을 불러오고, 다운로드
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms
)

# DataLoader를 사용해 학습 데이터를 배치로 묶어 불러오기
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# DataLoader를 사용해 테스트 데이터를 배치로 묶어 불러오기
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# GPU가 사용 가능한지 확인한 후, 디바이스 설정 (CUDA 또는 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN 모델을 생성하고, GPU 또는 CPU에 할당
model  = SimpleCNNClassifier().to(device)

# 손실 함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()  # 다중 클래스 분류에 적합한 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam 옵티마이저, 학습률 0.001

# 10번의 에포크 동안 학습
for epoch in range(10):

    # 모델을 학습 모드로 설정
    model.train()
    for images, labels in train_loader:
        # 데이터를 GPU 또는 CPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # 모델에 이미지를 입력하고 예측값 출력
        outputs = model(images)

        # 출력에서 가장 높은 값을 가진 클래스를 예측 (argmax)
        predicted = torch.max(outputs, 1)[1]

        # 예측과 실제 값 사이의 손실 계산
        loss = criterion(outputs, labels)

        # 옵티마이저의 기울기를 초기화하고 역전파를 통해 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()  # 손실에 대해 역전파
        optimizer.step()  # 가중치 업데이트

    # 모델을 평가 모드로 설정
    model.eval()
    test_loss = 0.0  # 테스트 손실 초기화
    correct = 0  # 맞춘 개수 초기화

    # 평가 중에는 기울기를 계산하지 않도록 설정 (메모리 효율화)
    with torch.no_grad():
        for images, labels in test_loader:
            # 데이터를 GPU 또는 CPU로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 모델에 이미지를 입력하고 예측값 출력
            outputs = model(images)

            # 출력에서 가장 높은 값을 가진 클래스를 예측
            predicted = torch.max(outputs, 1)[1]

            # 예측과 실제 값 사이의 손실 계산
            loss = criterion(outputs, labels)

            # 테스트 손실 누적
            test_loss += loss.item()

            # 맞춘 개수 누적
            correct += (labels == predicted).sum()

    # 에포크별로 테스트 손실과 정확도 출력
    print(
        f"epoch {epoch+1} - test loss: {test_loss / len(test_loader):.4f}, accuracy: {correct / len(test_data):.4f}"
    )