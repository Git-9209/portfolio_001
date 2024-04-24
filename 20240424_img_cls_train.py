import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

# 사용자 정의 데이터셋의 경로를 정의합니다
custom_trainset_path = r'D:/Project/Dataset/dog_and_cat/training_set'

# 데이터 변환을 정의합니다
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기를 사전 훈련된 모델의 입력 크기에 맞게 조정합니다
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 통계를 사용하여 정규화합니다
    # transforms.Normalize((R채널 평균, G채널 평균, B채널 평균), (R채널 표준편차, G채널 표준편차, B채널 표준편차))
])

# 사용자 정의 데이터셋을 생성합니다
train_dataset = datasets.ImageFolder(root=custom_trainset_path, transform=transform)

# 데이터 로더를 생성합니다
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 사전 훈련된 모델을 로드합니다 (예: ResNet18)
model = models.resnet18(pretrained=True)
# 출력 레이어를 사용자 정의 데이터셋의 클래스 수에 맞게 수정합니다
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

# GPU가 사용 가능한 경우 모델을 GPU로 이동합니다
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 옵티마이저를 정의합니다
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 훈련
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() #오차 역전파, 미분하여 손실함수에 끼친 영향력(변화량)을 구함
        optimizer.step() #손실함수를 최적화하도록 파라미터를 업데이트
        
    torch.save(model, './model/dog_and_cat_' + str(epoch + 1) + '.pt') # Model Save, 확장자 : *.pt
    print(f'epoch {epoch + 1}/{num_epochs}, loss: {loss.item()}')
