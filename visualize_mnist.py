import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# MNIST 데이터셋을 텐서로 변환
transform = transforms.ToTensor()

# 학습 데이터셋 다운로드 및 로드
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# 그래프 크기 설정
plt.rcParams["figure.figsize"] = (10, 12)

# 처음 100개의 이미지 시각화
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(train_data[i][0][0], cmap="gray")
    plt.title(train_data[i][1])
    plt.axis("off")

# 시각화 결과 보여주기
plt.show()