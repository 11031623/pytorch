import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 1. 数据处理 ==========
transform = transforms.Compose([
    transforms.Resize((32, 32)),      # GTSRB 图片尺寸不一，统一到 32×32
    transforms.ToTensor(),            # 转成张量，值范围 [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

BATCH_SIZE = 64

trainset = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")

# ========== 2. 构建 CNN 模型 ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 43)   # GTSRB 有 43 类

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 32×32 → 16×16
        x = self.pool(torch.relu(self.conv2(x)))   # 16×16 → 8×8
        x = self.pool(torch.relu(self.conv3(x)))   # 8×8 → 4×4
        x = x.view(x.size(0), -1)                  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)                        # 正则化
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)

# ========== 3. 损失函数与优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== 4. 训练与评估 ==========
num_epochs = 10
train_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    # ---- 训练 ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()    #判断正确的数量

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- 测试 ----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Test Acc: {test_acc:.2f}%")

# ========== 5. 可视化 ==========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cnn_result.png", dpi=150)
plt.show()