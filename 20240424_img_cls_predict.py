import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

# 사용자 정의 데이터셋의 경로를 정의합니다
custom_testset_path = r'D:/Project/Dataset/dog_and_cat/test_set'

# 데이터 변환을 정의합니다
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기를 사전 훈련된 모델의 입력 크기에 맞게 조정합니다
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 통계를 사용하여 정규화합니다
    # transofrms.Normalize((R채널 평균, G채널 평균, B채널 평균), (R채널 표준편차, G채널 표준편차, B채널 표준편차))
])

# 사용자 정의 데이터셋을 생성합니다
test_dataset = datasets.ImageFolder(root=custom_testset_path, transform=transform)

# 데이터 로더를 생성합니다
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # 사전 훈련된 모델을 로드합니다 (예: ResNet18)
# model = models.resnet18(pretrained=True)
# # 출력 레이어를 사용자 정의 데이터셋의 클래스 수에 맞게 수정합니다
# model.fc = nn.Linear(model.fc.in_features, len(custom_dataset.classes))

# GPU가 사용 가능한 경우 모델을 GPU로 이동합니다
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# 손실 함수와 옵티마이저를 정의합니다
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 테스트 map_location=device
model = torch.load('./model/dog_and_cat_best.pt') # Model Load
# CPU에서 학습한 모델을 GPU에서 불러올 때는 torch.load() 함수의 map_location 인자에 cuda:device_id 을 설정
model.eval()
correct, total, i = 0, 0, 1
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) # 각 열(1)마다 최댓값의 위치를 예측값으로 사용하겠다는 의미
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(i, len(test_loader), labels)
        i+=1

ACC = correct / total
print(f'테스트 정확도: {100 * ACC:.2f}%')
