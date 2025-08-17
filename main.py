import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

Base_dir = os.path.dirname(__file__)
model_dir = os.path.join(Base_dir, "Model")
Data_path= os.path.join(Base_dir, "Data")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "fashion_cnn.pth")
os.makedirs(Data_path, exist_ok=True)


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
)


train_dataset=torchvision.datasets.FashionMNIST(
    root=Data_path,
    train=True,
    download=True,
    transform=transform
)


test_dataset=torchvision.datasets.FashionMNIST(
    root=Data_path,
    train=False,
    download=True,
    transform=transform
)

train_loader=torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader=torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

print("Train batches:", len(train_loader), "Test batches:", len(test_loader))


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2=nn.Conv2d(32,64, kernel_size=3,padding=1)
        self.fc1=nn.Linear(64*7*7, 128)
        self.fc2=nn.Linear(128, 10)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/50], Loss: {running_loss/len(train_loader):.4f}")


correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


torch.save(model.state_dict(), model_path)

# Class names
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)


outputs = model(images)
_, predicted = torch.max(outputs, 1)


idx = 0
print("True label:", classes[labels[idx]])
print("Predicted :", classes[predicted[idx]])



import matplotlib.pyplot as plt
plt.imshow(images[idx].cpu().squeeze(), cmap="gray")
plt.title(f"True: {classes[labels[idx]]}, Pred: {classes[predicted[idx]]}")
plt.show()
