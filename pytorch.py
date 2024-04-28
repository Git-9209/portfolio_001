import glob
import torch
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class catdogDataset(Dataset):
    def __init__(self, path): # train=True, transform=None
        self.path = path
        # if train:
        #     self.cat_path = path + '/training_set/cats'
        #     self.dog_path = path + '/training_set/dogs'
        # else:
        #     self.cat_path = path + '/test_set/cats'
        #     self.dog_path = path + '/test_set/dogs'
        #
        # self.cat_img_list = glob.glob(self.cat_path + '/*.png')
        # self.dog_img_list = glob.glob(self.dog_path + '/*.png')
        #
        # self.transform = transform
        #
        # self.img_list = self.cat_img_list + self.dog_img_list
        # self.class_list = [0] * len(self.cat_img_list) + [1] * len(self.dog_img_list)
    
    def __len__(self):
        return self.path
        # return len(self.img_list)
    
    def __getitem__(self, idx):
        return torch.LongTensor([idx])
        # img_path = self.img_list[idx]
        # label = self.class_list[idx]
        # img = Image.open(img_path)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # return img, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = catdogDataset(path='D:/Project/Dataset/dog_and_cat', train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, drop_last=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 5
    for epoch in range(num_epochs):
        for img, label in dataloader:
            inputs, labels = img.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(img.size(), label)
        print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item()}')
